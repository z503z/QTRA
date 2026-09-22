# Third-party notices

This is a private release candidate. A project-wide license for the authors'
new code has not yet been assigned. The following notices do not replace the
terms of any upstream component, dataset or pretrained model.

| Component | Source / provenance | Notice |
|---|---|---|
| QAG dataset reader and the surrounding CLIP integration lineage | [VisTA](https://github.com/like413/VisTA); reader retained, model imports reorganized | [CC BY-NC 4.0](licenses/VisTA-CC-BY-NC-4.0.txt) as provided by upstream |
| CLIP implementation, tokenizer and BPE vocabulary | [OpenAI CLIP](https://github.com/openai/CLIP) | [MIT](licenses/CLIP-MIT.txt) |
| Modified CLIP implementation used by VisTA | [CRIS](https://github.com/DerrickWang005/CRIS.pytorch) | [MIT](licenses/CRIS-MIT.txt); retained in `qtra/models/clip.py` |
| X3D implementation | [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) installed as a dependency | [Apache 2.0](licenses/PyTorchVideo-Apache-2.0.txt) |
| Perception-frame temporal modeling formulation | [Change3D](https://github.com/zhuduowang/Change3D) | Scientific attribution; the release X3D file is a small configuration wrapper over PyTorchVideo, not the modified upstream source file |

The refactoring changes import paths, removes unused experimental architectures,
adds explicit stage configurations, and separates training/validation from
standalone evaluation. Forward computations of the released QAG model were
checked against the original checkpoint implementation.

QAG-360K's owners prohibit redistribution of their dataset in their repository
instructions. No dataset is included here. Obtain QAG-360K, CDVQA and SECOND from
their respective owners and follow each dataset's terms. Download pretrained
CLIP/X3D weights from their official sources and follow the associated terms.
