from types import SimpleNamespace

import recurrent.impls.video_memory as video_memory


def test_missing_lmdb_disables_optional_disk_cache(monkeypatch):
    monkeypatch.setattr(video_memory, "lmdb", None)
    monkeypatch.setattr(video_memory.RDataset, "__init__", lambda self, **kwargs: None)

    dataset = video_memory.VideoMemoryDataset(
        recurrent_config=SimpleNamespace(
            max_video_clips=4,
            video_clip_token_size=1,
            video_key="video",
            video_root="",
            max_video_frame=1,
        ),
        data_files=[],
        tokenizer=object(),
        data_config=SimpleNamespace(truncation="center"),
        processor=object(),
        use_cache=True,
    )

    assert dataset.use_cache is False
