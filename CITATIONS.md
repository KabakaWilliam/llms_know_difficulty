# Citations

If you use PIKA or our pre-trained probes, please cite our work.

## Primary Citation

**LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations** (2026)

```bibtex
@misc{lugoloobi_llms_2026,
    title = {{LLMs} {Encode} {Their} {Failures}: {Predicting} {Success} from {Pre}-{Generation} {Activations}},
    shorttitle = {{LLMs} {Encode} {Their} {Failures}},
    url = {http://arxiv.org/abs/2602.09924},
    doi = {10.48550/arXiv.2602.09924},
    abstract = {Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty.},
    publisher = {arXiv},
    author = {Lugoloobi, William and Foster, Thomas and Bankes, William and Russell, Chris},
    month = feb,
    year = {2026},
    note = {arXiv:2602.09924 [cs]},
    keywords = {Computer Science - Artificial Intelligence, Computer Science - Computation and Language, Computer Science - Machine Learning},
}
```

## Earlier Work

**LLMs Encode How Difficult Problems Are** (2025)

```bibtex
@misc{lugoloobi_llms_2025,
    title = {{LLMs} {Encode} {How} {Difficult} {Problems} {Are}},
    url = {http://arxiv.org/abs/2510.18147},
    doi = {10.48550/arXiv.2510.18147},
    publisher = {arXiv},
    author = {Lugoloobi, William and Russell, Chris},
    month = oct,
    year = {2025},
    note = {arXiv:2510.18147 [cs]},
}
```

## Links

- **Paper (2026)**: [arXiv:2602.09924](https://arxiv.org/abs/2602.09924)
- **Paper (2025)**: [arXiv:2510.18147](https://arxiv.org/abs/2510.18147)
- **Code**: [github.com/KabakaWilliam/llms_know_difficulty](https://github.com/KabakaWilliam/llms_know_difficulty)
- **Pre-trained Probes**: [huggingface.co/CoffeeGitta/pika-probes](https://huggingface.co/CoffeeGitta/pika-probes)
