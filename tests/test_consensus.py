from src.orchestration.consensus import calculate_similarity


def test_similarity_is_one_for_single_text() -> None:
    assert calculate_similarity(["only one"]) == 1.0


def test_similarity_returns_reasonable_average() -> None:
    score = calculate_similarity(
        [
            "tcp handshake uses syn syn-ack ack",
            "a tcp connection starts with syn then syn-ack then ack",
        ]
    )

    assert 0.0 <= score <= 1.0
    assert score > 0.0
