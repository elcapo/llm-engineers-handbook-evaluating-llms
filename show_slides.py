import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", layout_file="layouts/show_slides.slides.json")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # LLM Engineer's Handbook

    ## Chapter 7: **Evaluating LLMs**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # In previous chapters

    * We **finetuned** a Llama model on some articles written by the authors of the book
    * We applied **direct preference optimization** to teach if to follow the desired writing style
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Introduction

    * There is no unified approach to measuring a model's performance
    * But there are patterns, recipes and standard benchmarks
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # In this chapter

    We'll cover the following topics:

    * Model evaluation
    * RAG evaluation
    * Evaluating our model
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Comparing ML and LLM evaluation

    ## Machine Learning

    * Models typically process and produce structured data
    * Their evaluation is about how accurate and efficient models are at that

    ## Large Language Models

    * Models process and produce unstructured data
    * Their evaluation is about how well they understand and generate language
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Comparing ML and LLM evaluation

    ## Differences

    * In **ML** features (and target variables) have to be prepared in a specific way while **LLMs** work with raw data
    * In **ML** we typically use objective performance metrics: accuracy, precision, recall, or mean squared error but as **LLMs** can handle multiple tasks this is not possible
    * In **ML** being able to interpret why a model made a certain prediction is a core part of the evaluation but direct interpretation is not possible with **LLMs**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # General Purpose LLM Evaluations

    ## Pretraining

    * Training and validation losses: **cross-entropy** used to measure the difference between the predicted and the target distribution
    * **Perplexity**: an exponential form of cross-entropy that measures the model "surprise"
    * **Gradient norm**: measures gradients during training to detect potential instabilities
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # General Purpose LLM Evaluations

    ## After Pretraining

    * [MMLU](https://arxiv.org/pdf/2009.03300): general knowledge questions across 57 subjects
    * [HellaSwag](https://arxiv.org/abs/1905.07830) and [HellaSwag-Pro](https://arxiv.org/abs/2502.11393): complete situations with the most likely output from a list of multiple choices
    * [ARC-C](https://arxiv.org/abs/1803.05457): multiple-choice science questions that require causal reasoning
    * [Winogrande](https://arxiv.org/pdf/1907.10641): common sense reasoning through carefully crafted sentences
    * [PIQA](https://arxiv.org/abs/1911.11641): Physical commonsense understanding
    * [ARC-AGI-3](https://arcprize.org): puzzle based scenarios that require logical and spacial reasining
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # General Purpose LLM Evaluations

    ## After Finetuning

    * [IFEval](https://arxiv.org/abs/2311.07911): tests the ability to follow instructions
    * [Chatbot Arena](https://arxiv.org/pdf/2403.04132): framework where humans vote for the best answer
    * [AlpacaEval](https://arxiv.org/abs/2404.04475): automatic evaluation of finetuned models highly correlated with the Chatbot Arena
    * [MT-Bench-101](https://arxiv.org/abs/2402.14762): evaluation based in multi-turn conversations
    * [GAIA](https://arxiv.org/abs/2311.12983): tests abilities like tool usage in a multi-step fashion
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # General Purpose LLM Evaluations

    ## Final Considerations

    > "Public benchmarks can be gamed by training models on test data or samples that are very similar to benchmark datasets."
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Domain Specific LLM Evaluations

    ## Domain Based Evaluations

    * [Open Medical-LLM Leaderboard](https://huggingface.co/blog/leaderboard-medicalllm): tests performance in medical question-answering tasks
    * [BigCodeBench Leaderboard](https://bigcode-bench.github.io): tests performance in code generation (and completion)
    * [Hallucinations Leaderboard](https://arxiv.org/abs/2404.05904): tests the tendendy to produce false or unsupported information
    * [Enterprise Scenarios Leaderboard](https://huggingface.co/blog/leaderboard-patronus): evaluates the performance in enterprise use cases

    ## Language Based Evaluations

    * [OpenKO-LLM Leaderboard](https://arxiv.org/abs/2405.20574): evaluates an LLM performance in Korean
    * [Open Portuguese LLM Leaderboard](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard): evaluates an LLM performance in Portuguese
    * [Open Arabic LLM Leaderboard](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard): evaluates an LLM performance in Arabic
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Task Specific LLM Evaluations

    ## Summarization

    * [Recall-Oriented Understudy for Gisting Evaluation](https://en.wikipedia.org/wiki/ROUGE_(metric)): measures the overlap between the generated and reference text using n-grams

    ## Classification

    * [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision): "overall correctness across all classes by calculat-
    ing the proportion of correct predictions among the total number of cases
    examined"
    * [Precision](https://en.wikipedia.org/wiki/Precision_and_recall): "Out of all the cases indicated as positive, how many are as actual positives?"
    * [Recall](https://en.wikipedia.org/wiki/Precision_and_recall): "Of all the positive cases, how many did our model correctly
    identify?"
    * [F1 Score](https://en.wikipedia.org/wiki/F-score): "1-score is critical when we care equally about avoiding false positives (e.g., incorrectly marking an important email as spam) and false negatives (e.g., missing a spam email)."

    Quotes are from: [The Little Book of ML Metrics](https://www.nannyml.com/metrics)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Task Specific LLM Evaluations

    When we cannot fallback to ML metrics, we can create our own. Our main choices are:

    * Making the model answer questions and comparing the answers with their known correct answers (typically involves limiting the format of the output)
    * Checking the model's predicted probabilities so that we can get an idea of how confident the model is in its answer

    The authors recommend starting with the first approach for its simplicity
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Task Specific LLM Evaluations

    ## LLM as a judge

    If we ask our model to generate answers and the answers are not choice-based, we can use another model to evaluate them

    But judge LLMs have biases:

    * They tend to favor assertive and verbose answers
    * They tend to overrate confidenr answers (even when they are less accurate)
    * They can prefer some writing styles (even when they are not related with the answer quality)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # RAG Evaluation

    We are not evaluating a model anymore but a pipeline. We have to measure:

    * **Retrieval accuracy**: are the retrieved documents the ones that are most relevant for the task?
    * **Integration quality**: difference between the responses with and without context
    * **Factuality and relevance**: whether the output is relevant and based in the retrieved documents
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The Ragas Framework

    ## Strengths

    * [Open-source](https://docs.ragas.io/en/stable/getstarted/), easy to integrate with Python workflows
    * Provides multiple metrics (faithfulness, answer relevancy, context precision/recall)
    * Designed for LLM-based evaluation → leverages models to judge quality
    * Flexible: can be extended with custom evaluators

    ## Weaknesses

    * Heavy reliance on LLMs for scoring (possible bias or inconsistency)
    * Requires careful prompt design for evaluation consistency
    * Can be slower and more costly for large-scale evaluations
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The ARES Framekwork

    ## Strengths

    * [Open-source](https://ares-ai.vercel.app)
    * Focuses specifically on retrieval evaluation (before generation)
    * Model-agnostic: does not rely on LLMs for scoring
    * Provides clear, reproducible metrics (recall, precision, F1)
    * Scales well for large datasets

    ## Weaknesses

    * Limited to retrieval performance → does not evaluate generated answers
    * Less flexible compared to LLM-based evaluators
    * May miss subtle aspects of answer quality or context use
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Ragas vs. ARES

    * **Ragas** is an end-to-end RAG evaluation (retrieval + generation)
    * **Ragas** is rich, flexible but more complex and costly
    * **ARES** is a focused, lightweight and efficient for retrieval only
    * **ARES** is simpler, but lacks answer quality assessment

    ## Conclusion

    * Use **ARES** when the priority is retrieval system performance
    * Use **Ragas** when evaluating the full RAG pipeline, including answer quality
    * Combine both for a holistic evaluation
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Evaluting the TwinLlama Model

    ## Goal

    We want to evaluate this three models:

    * Reference model: [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
    * Finetuned model: [mlabonne/TwinLlama-3.1-8B-GGUF](https://huggingface.co/mlabonne/TwinLlama-3.1-8B-GGUF)
    * DPO model: [mlabonne/TwinLlama-3.1-8B-DPO-GGUF](https://huggingface.co/mlabonne/TwinLlama-3.1-8B-DPO-GGUF)

    Using the **instructions** from this dataset:

    * Instructions dataset: [mlabonne/llmtwin](https://huggingface.co/datasets/mlabonne/llmtwin)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # LLM as a judge

    We'll use OpenAI's **gpt-4o-mini** as a judge who will follow the [evaluate-answer](./prompts/evaluate-answer.md) prompt.

    ```
    You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:

    1. Accuracy: How factually correct is the information presented in the answer? You are the technical expert in this topic.
    2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

    Accuracy scale:

    1 (Poor): Contains factual errors or misleading information
    2 (Good): Modtly accurate with minor errors or omissions
    3 (Excellent): Highly accurate and comprehensive

    Style scale:

    1 (Poor): Too formal, uses some overly complex words
    2 (Good): Good balance of technical content and accesibility, but still uses formal words and expressions
    3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

    Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor of its predecessor, the original Llama architecture.

    Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.

    Instruction: {instruction}

    Answer: {answer}
    ```
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
