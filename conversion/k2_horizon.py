from __future__ import annotations

from pathlib import Path

from .base import ModelBase, TextModel, gguf


@ModelBase.register("K2HorizonForCausalLM")
@ModelBase.example("IFM/K2-Horizon-7B")
class K2HorizonModel(TextModel):
    """Dense K2-Horizon-7B converter.

    The 7B config is dense (num_experts=0, mova_num_experts=0). Keep the
    converter narrow until separate MoE/MoVA weights and graph semantics are
    pinned.
    """

    model_arch = gguf.MODEL_ARCH.K2HORIZON

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        from transformers import PreTrainedTokenizerFast

        # TokenizersBackend is not registered by older Transformers releases.
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.dir_model / "tokenizer.json"))
        return super().get_vocab_base(tokenizer)

    def set_vocab(self):
        super().set_vocab()

        template_path = (
            Path(__file__).parent.parent
            / "models"
            / "templates"
            / "k2-horizon.jinja"
        )
        template = template_path.read_text(encoding="utf-8")
        self.gguf_writer.remove_key(gguf.Keys.Tokenizer.CHAT_TEMPLATE)
        self.gguf_writer.add_chat_template(template)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        if int(self.hparams.get("num_experts", 0)) != 0:
            raise ValueError("K2-Horizon-7B port requires num_experts=0")
        if int(self.hparams.get("mova_num_experts", 0)) != 0:
            raise ValueError("K2-Horizon-7B port requires mova_num_experts=0")

        norm_groups = int(self.hparams.get("layernorm_num_groups", 1))
        self.gguf_writer.add_group_norm_groups(norm_groups)

        rope_head_dim = self.hparams.get("rope_head_dim")
        if rope_head_dim is not None:
            self.gguf_writer.add_rope_dimension_count(int(rope_head_dim))
