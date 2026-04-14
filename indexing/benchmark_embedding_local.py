import argparse
import statistics
import time
from dataclasses import dataclass
from huggingface_hub import snapshot_download
import torch
from sentence_transformers import SentenceTransformer
import os
os.environ["HF_HUB_OFFLINE"] = "1"

# Sample 2 minute podcast transcript chunks
TEXTS = [
    "Welcome, welcome everyone to MCU re-watch your path to Avengers end game. I am your host Mark Turca and on this episode. We are talking about doctor strange and to do so I am joined by my fellow co-hosts here first up. Mr. Paris Lily Paris. What's up, man? Hey, what's going on sorcerer Supreme. I was Doctor Strange ignorant going into this movie never read a comic anything. So this was a good education for me. Kurt I think you know, I think for many people it was definitely almost like a man you're a lot of people's Heroes you kind of heard about but don't know anything about I think for many this was their first foray into doctor strange and our fellow co-host here Swain Swain, what's up, man? Hello. I am an actor strange is a nice change of pace after all these weeks of really intense Avengers and Civil War and all that. It's Like it's a it's a very nice palate cleanser for sure. What was your knowledge of dr. strange going into this that you have? I know my knowledge was very little but when I find out that something is happening. I binge it almost to know everything about it going into it, which is the opposite of Kelly. That's not what she does. She'll be like, I just don't want to know anything about it. You could tell me after and I'll be like, okay, here's where everything connects into the car. Sorry thing well for me actually Doctor Strange was the very first comic book that I went to a comic shop and bought with my very own money was a Ghost Rider comic I remember it was fourth grade was a ghost right? Actually, it's hanging on the wall of those you ever seen my room here. I've Comics behind me Ghost Rider issue 12. It was like the Danny Ketch Ghost Rider and it was a doctor strange issue. He was I don't think he's the villain. I haven't read the book in a long time. But to me, it's hanging on my wall. But that was my one of my first comic are was my first Comic book purchase had",
    "which is quite a big deal. It's a big deal. I know but the thing is all we went to can all the Canlis pipe. I can divide by candlelight and windy. Doubt she was doing she came here and she was so excited about winter and I was like, well, yeah. Yeah, I'm on just wait for the real cold and dark and wet but she was so sad for snow and snow comes and she's excited and later. I hear that she is in the hospital. Oh what happened to her? I'm broke my God first time you went sledding. Well, yeah. It was in the middle of the night. Don't go sledding in the middle and there was a lot of rocks and trees on the like nachos of Hill but like big boulder rocks like Granite. Do you really make herself look like just stupid right now. I was with a friend. Okay, we had a really fun time for like we were out there for like half an hour and then until we almost went back and then I like, oh, let's go one more time and then that one more time. I might have hit a rock might have yeah, and she's not like she has had any complications. I've had a lot of the complications. I just had surgery recently to take out the metal that they put in the first surgery. Yeah, so now it's actually a lot better and I can move I had pain every day forth like the whole time here for years. Yeah. So so that's like a part of the story, but Julie is a champ. She's great. So coming to like Do missions. Of course you you don't do missions because you love coffee and you don't even love coffee. I don't really love coffee not exactly. So you're not coming to Sweden to like preach about a cop because you guys already coffin eyes. There's I like what the number we're two or three in the country in the world of coffee consumption.",
]

MODEL_PATH = "../../../Embed_Models/"

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "cointegrated/rubert-tiny2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-small",
    "microsoft/harrier-oss-v1-270m",
    "google/embeddinggemma-300m",  # Requires accepting terms on Hugging Face and authentication
    "voyageai/voyage-4-nano",
    "mixedbread-ai/mxbai-embed-large-v1",
]

RUNS = 10  # Number of timed encoding runs per model and text
WARMUP = 3  # Number of warmup runs before timing


@dataclass
class BenchmarkResult:
    model_name: str
    avg_encode_seconds: float
    min_encode_seconds: float
    max_encode_seconds: float
    embedding_shape: tuple[int, ...]
    param_count: int = 0


def _sync_if_needed(device: str):
    """Synchronize CUDA if using GPU to ensure accurate timing."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _count_parameters(model) -> int:
    """Count total parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def benchmark_model(
    model_name: str,
    model_path: str,
    runs: int,
    warmup: int,
    device: str,
) -> BenchmarkResult:
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    # model = model.half()   ## Quantize to fp16

    # warmup runs to stabilize performance
    for _ in range(warmup):
        model.encode(TEXTS[0], convert_to_numpy=True)

    run_times: list[float] = []
    embedding_shape: tuple[int, ...] = tuple()
    for _ in range(runs):
        for text in TEXTS:
            _sync_if_needed(device)
            start = time.perf_counter()
            embedding = model.encode(text, convert_to_numpy=True)
            _sync_if_needed(device)
            elapsed = time.perf_counter() - start
            run_times.append(elapsed)

            if not embedding_shape:
                embedding_shape = tuple(int(dim) for dim in embedding.shape)

    avg_encode = statistics.mean(run_times)
    param_count = _count_parameters(model)

    return BenchmarkResult(
        model_name=model_name,
        avg_encode_seconds=avg_encode,
        min_encode_seconds=min(run_times),
        max_encode_seconds=max(run_times),
        embedding_shape=embedding_shape,
        param_count=param_count,
    )


def _format_param_count(count: int) -> str:
    """Format parameter count with M/B suffixes."""
    if count == 0:
        return "N/A"
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    else:
        return f"{count / 1e3:.2f}K"


def print_summary(results: list[BenchmarkResult]) -> None:
    print("\nEmbedding Benchmark Results")
    print("=" * 135)
    print(
        f"{'Model':60} {'Params':>12} {'AvgEncode(ms)':>13} {'Min(ms)':>10} {'Max(ms)':>10} {'Shape':>14}"
    )
    print("-" * 135)

    for result in sorted(results, key=lambda r: r.avg_encode_seconds):
        print(
            f"{result.model_name:60} "
            f"{_format_param_count(result.param_count):>12} "
            f"{result.avg_encode_seconds * 1000:13.1f} "
            f"{result.min_encode_seconds * 1000:10.1f} "
            f"{result.max_encode_seconds * 1000:10.1f} "
            f"{str(result.embedding_shape):>14} "
        )

    print("-" * 135)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sentence embedding models.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run inference on.",
    )
    return parser.parse_args()

def parse_model_list():
    model_list = []
    for models in os.listdir(MODEL_PATH):
        tmp_path = MODEL_PATH+models+"/snapshots/"
        final_pth = os.listdir(tmp_path)
        if len(final_pth) == 1:
            final_model = tmp_path+final_pth[0]
            model_list.append(final_model)
    return model_list

def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    avg_text_len = sum(map(len, TEXTS)) / len(TEXTS)
    print(f"Average Text length: {avg_text_len} characters")
    print(f"Warmup runs: {WARMUP}, timed runs per text: {RUNS}\n")

    results: list[BenchmarkResult] = []

    failed_count = 0
    for model_name in MODELS:
        print(f"\nBenchmarking: {model_name}")
        try:
            model_path = snapshot_download(
                    model_name,
                    local_files_only=True
                )
            
            result = benchmark_model(
                model_name=model_name,
                model_path=model_path,
                runs=RUNS,
                warmup=WARMUP,
                device=device,
            )
            results.append(result)
        except Exception as e:
            failed_count += 1
            print(f"  FAILED: {e}")

    print_summary(results)
    print(
        f"\nCompleted benchmarking {len(results)} models with {failed_count} failures."
    )


if __name__ == "__main__":
    main()
