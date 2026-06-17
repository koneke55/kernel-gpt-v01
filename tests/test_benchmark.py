from kernel_gpt.benchmark.runner import BenchmarkRunner


def test_runner_dummy():
    runner = BenchmarkRunner()
    res = runner.run(None, (), ())
    assert res.correctness
