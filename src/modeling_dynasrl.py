import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


# =========================================================================
# Module 1: Schema Mapper (Projector)
# Function: Corresponds to Mapping Network M_phi in paper Eq (4).
# Task: Translate aggregated Schema semantic vector (E_S) into virtual Token sequence (V_S) understandable by LLM.
# =========================================================================
class DynaSRLProjector(nn.Module):
    """
    Lightweight MLP: Maps aggregated Schema Embedding to Latent Vectors
    Structure: Linear -> ReLU -> Linear
    """

    def __init__(self, hidden_size, latent_len=10):
        super().__init__()
        # [Design Detail] Why *2?
        # First *2 (Feature Expansion): Expands feature dimension to give model more capacity for complex Schema semantics.
        # Second *2: Matches output dimension of previous layer.
        # Final output dimension is hidden_size * latent_len, as we need to generate latent_len vectors of size hidden_size.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),  # Introduce non-linearity
            nn.Linear(hidden_size * 2, hidden_size * latent_len)
        )
        self.latent_len = latent_len  # L: Number of virtual tokens
        self.hidden_size = hidden_size  # d: Hidden dimension of LLM

    def forward(self, schema_embeds):
        # schema_embeds: [Batch, Hidden_Size] (Aggregated global Schema representation E_S)

        batch_size = schema_embeds.shape[0]

        # 1. MLP Projection
        # Input: [Batch, d] -> Output: [Batch, L * d]
        latent_vec = self.mlp(schema_embeds)

        # 2. Reshape
        # Slice into L independent vectors.
        # Output: [Batch, L, d] -> This is V_S
        return latent_vec.view(batch_size, self.latent_len, self.hidden_size)


# =========================================================================
# Module 2: DynaSRL Main Model Architecture
# Function: Integrates Projector and Base LLM for full forward pass from "Schema + Text" to "Role Labeling".
# =========================================================================
class DynaSRLModel(nn.Module):
    def __init__(self, base_model_path, latent_len=10, use_lora=True, use_projector=True):
        super().__init__()
        print(f"Loading Base Model from: {base_model_path}")

        # 2.1 Load Base Model
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )

        # 2.2 Initialize Projector - Decide whether to initialize based on switch
        self.hidden_size = self.llm.config.hidden_size
        if use_projector:
            self.projector = DynaSRLProjector(self.hidden_size, latent_len).to(self.llm.device).to(torch.bfloat16)
        else:
            self.projector = None

        # 2.3 Configure LoRA (Phase 1 & Phase 2 Core)
        # Corresponds to Regularized Dual-Adaptation in paper.
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=64,  # [Critical Param] Rank: Large (64) ensures strong fitting for specific domain (SRL).
                lora_alpha=128,  # Alpha usually 2x Rank.
                lora_dropout=0.05,
                # Inject LoRA into all linear layers of Qwen.
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()

        # 2.4 Disable cache
        self.llm.config.use_cache = False

        # 2.5 Hugging Face compatibility hacks
        # When Trainer's load_best_model_at_end=True, it expects these attributes
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Redirect to the base LLM's gradient checkpointing
        :param gradient_checkpointing_kwargs: Arguments for gradient checkpointing
        :return: None
        """
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, attention_mask, labels, schema_input_ids_list=None):
        """
        Core Forward Propagation Logic.
        Corresponds to paper Eq (5): I(X, S) = V_S (+) I_meta (+) Linearize(S) (+) X
        :param input_ids: Input token IDs for text
        :param attention_mask: Attention mask
        :param labels: Labels for loss calculation
        :param schema_input_ids_list: List of schema definition token IDs
        :return: Model outputs
        """
        # Logic: If no Projector, use standard LLM forward
        if self.projector is None:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True
            )

        batch_size = input_ids.shape[0]
        latent_prefixes = []

        # Get LLM Embedding Layer (Token ID to Vector)
        embed_layer = self.llm.get_input_embeddings()

        # =====================================================================
        # Step A: Generate Schema Latent Vectors V_S (Schema-Conditioned Latent Initialization)
        # =====================================================================
        for b in range(batch_size):
            # s_ids: [Num_Roles, Seq_Len]
            s_ids = schema_input_ids_list[b].to(self.llm.device)

            # [Design Detail] torch.no_grad():
            # Paper setting: Schema Embedding is based on "frozen" LLM parameters.
            with torch.no_grad():
                s_embeds = embed_layer(s_ids)  # [Num_Roles, Seq_Len, d]

                # Eq (4): Dual Mean Aggregation
                # 1. Role-level Mean: Compress sequence dim (Seq_Len -> 1)
                s_role_repr = s_embeds.mean(dim=1)  # [Num_Roles, d]
                # 2. Global Mean: Compress role dim (Num_Roles -> 1)
                global_schema_repr = s_role_repr.mean(dim=0, keepdim=True)  # [1, d]

            # Map to virtual instruction vector V_S via Projector
            # [1, d] -> [1, L, d]
            v_s = self.projector(global_schema_repr)
            latent_prefixes.append(v_s)

        # Stack list to Batch Tensor: [Batch, L, d]
        latent_prefixes = torch.cat(latent_prefixes, dim=0)

        # =====================================================================
        # Step B: Concatenate Full Input
        # =====================================================================
        # 1. Vectorize text part: I_meta + Linearize(S) + X
        inputs_embeds = embed_layer(input_ids)  # [Batch, Seq_Len_Text, d]

        # 2. Concatenate: [Latent Vector V_S] + [Text Vector]
        # Corresponds to (+) operation in paper
        combined_embeds = torch.cat([latent_prefixes, inputs_embeds], dim=1)

        # =====================================================================
        # Step C: Align Mask and Labels
        # =====================================================================
        # Input length increased by L, update Attention Mask.
        prefix_len = latent_prefixes.shape[1]  # L

        # 1. Extend Attention Mask
        prefix_mask = torch.ones((batch_size, prefix_len), device=attention_mask.device)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 2. Extend Labels
        # [Design Detail] -100 Masking: Latent vectors are input, not output.
        prefix_labels = torch.full((batch_size, prefix_len), -100, device=labels.device)
        combined_labels = torch.cat([prefix_labels, labels], dim=1)

        # =====================================================================
        # Step D: LLM Calculation
        # =====================================================================
        # Call Hugging Face standard interface using inputs_embeds
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            use_cache=False,
            return_dict=True
        )
        return outputs
