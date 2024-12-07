from typing import List, Tuple, Dict, Optional
import pytest
import numpy as np
from torch.utils.data import DataLoader

from ..ucf_dataset import UCFDataSet, SubsetSampler, LOCAL_UCF_PATH


@pytest.mark.unittest
@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("video_size", [(720, 480), (320, 240)])
def test_ucf_dataset(split, video_size):

    dataset = UCFDataSet(path=LOCAL_UCF_PATH,split=split, size=video_size)

    sample_0 = dataset[0]

    sample_1 = dataset[1]

    assert isinstance(sample_0[1], np.ndarray)
    assert sample_0[1].shape[-2:] == video_size
    assert sample_1[1].shape[-2:] == video_size



@pytest.mark.unittest
@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("video_size", [(720, 480)])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_ucf_dataloader(split, video_size, batch_size):

    ucf_dataset = UCFDataSet(path=LOCAL_UCF_PATH,split=split, size=video_size)

    dataloader = DataLoader(
        ucf_dataset, batch_size=batch_size,
        pin_memory=False, num_workers=4,
        shuffle=True)
    sample_0 = next(iter(dataloader))

    sample_0_prompt, sample_0_video = sample_0[0], sample_0[1]

    assert sample_0[1].shape[-2:] == video_size
    assert len(sample_0_prompt) == batch_size
    print("prompts: ", sample_0_prompt)
    print("videos: ", sample_0_video.shape)



@pytest.mark.unittest
@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_ucf_dataloader_smapler(split, batch_size):

    ucf_dataset = UCFDataSet(path=LOCAL_UCF_PATH,split=split, size=(720, 480))
    sampler = SubsetSampler(ucf_dataset, subset_size=100)

    dataloader = DataLoader(
        ucf_dataset, batch_size=batch_size,
        pin_memory=False, num_workers=4,
        shuffle=False, sampler=sampler)
    sample_0 = next(iter(dataloader))

    sample_0_prompt, sample_0_video = sample_0[0], sample_0[1]

    assert len(sample_0_prompt) == batch_size
    print("prompts: ", sample_0_prompt)
    print("videos: ", sample_0_video.shape)

    print("len(full_dataset)", len(ucf_dataset))
    print("len(subset_dataloader)", len(dataloader))