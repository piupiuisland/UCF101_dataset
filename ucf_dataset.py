import torch
import os
import numpy as np
import cv2

from typing import Any, Optional, Dict, Union, Tuple, List
from torch.utils.data import Dataset, DataLoader

LOCAL_UCF_PATH = "/home/ubuntu/work_root/llm_repo/text2video_eval/ucf101_dataset/"

UCF_LABELS = {
    "ApplyEyeMakeup" :	"Apply Eye Makeup",
    "ApplyLipstick" :	"Apply Lipstick",
    "Archery" :	"Archery",
    "BabyCrawling" :	"Baby Crawling",
    "BalanceBeam":	"Balance Beam",
    "BandMarching" : "Band Marching",
    "BaseballPitch" : "Baseball Pitch",
    "Basketball" : "Basketball",
    "BasketballDunk" : "Basketball Dunk",
    "BenchPress" : "Bench Press",
    "Biking" : "Biking",
    "Billiards" : "Billiards",
    "BlowDryHair" : "Blow Dry Hair",
    "BlowingCandles" : "Blowing Candles",
    "BodyWeightSquats" : "Body Weight Squats",
    "Bowling" : "Bowling",
    "BoxingPunchingBag" : "Boxing Punching Bag",
    "BoxingSpeedBag" : "Boxing Speed Bag",
    "BreastStroke" : "Breast Stroke",
    "BrushingTeeth" : "Brushing Teeth",
    "CleanAndJerk" : "Clean And Jerk",
    "CliffDiving" : "Cliff Diving",
    "CricketBowling" : "Cricket Bowling",
    "CricketShot" : "Cricket Shot",
    "CuttingInKitchen" : "Cutting In Kitchen",
    "Diving" : "Diving",
    "Drumming" : "Drumming",
    "Fencing" : "Fencing",
    "FieldHockeyPenalty" : "Field Hockey Penalty",
    "FloorGymnastics" : "Floor Gymnastics",
    "FrisbeeCatch" : "Frisbee Catch",
    "FrontCrawl" : "Front Crawl",
    "GolfSwing" : "Golf Swing",
    "Haircut" : "Haircut",
    "Hammering" : "Hammering",
    "HammerThrow" : "Hammer Throw",
    "HandstandPushups" : "Handstand Pushups",
    "HandstandWalking" : "Handstand Walking",
    "HeadMassage" : "Head Massage",
    "HighJump" : "High Jump",
    "HorseRace" : "Horse Race",
    "HorseRiding" : "Horse Riding",
    "HulaHoop" : "Hula Hoop",
    "IceDancing" : "Ice Dancing",
    "JavelinThrow" : "Javelin Throw",
    "JugglingBalls" : "Juggling Balls",
    "JumpingJack" : "Jumping Jack",
    "JumpRope" : "Jump Rope",
    "Kayaking" : "Kayaking",
    "Knitting" : "Knitting",
    "LongJump" : "Long Jump",
    "Lunges" : "Lunges",
    "MilitaryParade" : "Military Parade",
    "Mixing" : "Mixing",
    "MoppingFloor" : "Mopping Floor",
    "Nunchucks" : "Nunchucks",
    "ParallelBars" : "Parallel Bars",
    "PizzaTossing" : "Pizza Tossing",
    "PlayingCello" : "Playing Cello",
    "PlayingDaf" : "Playing Daf",
    "PlayingDhol" : "Playing Dhol",
    "PlayingFlute" : "Playing Flute",
    "PlayingGuitar" : "Playing Guitar",
    "PlayingPiano" : "Playing Piano",
    "PlayingSitar" : "Playing Sitar",
    "PlayingTabla" : "Playing Tabla",
    "PlayingViolin" : "Playing Violin",
    "PoleVault" : "Pole Vault",
    "PommelHorse" : "Pommel Horse",
    "PullUps" : "Pull Ups",
    "Punch" : "Punch",
    "PushUps" : "Push Ups",
    "Rafting" : "Rafting",
    "RockClimbingIndoor" : "Rock Climbing Indoor",
    "RopeClimbing" : "Rope Climbing",
    "Rowing" : "Rowing",
    "SalsaSpin" : "Salsa Spin",
    "ShavingBeard" : "Shaving Beard",
    "Shotput" : "Shotput",
    "SkateBoarding" : "Skate Boarding",
    "Skiing" : "Skiing",
    "Skijet" : "Skijet",
    "SkyDiving" : "Sky Diving",
    "SoccerJuggling" : "Soccer Juggling",
    "SoccerPenalty" : "Soccer Penalty",
    "StillRings" : "Still Rings",
    "SumoWrestling" : "Sumo Wrestling",
    "Surfing" : "Surfing",
    "Swing" : "Swing",
    "TableTennisShot" : "Table Tennis Shot",
    "TaiChi" : "Tai Chi",
    "TennisSwing" : "Tennis Swing",
    "ThrowDiscus" : "Throw Discus",
    "TrampolineJumping" : "Trampoline Jumping",
    "Typing" : "Typing",
    "UnevenBars" : "Uneven Bars",
    "VolleyballSpiking" : "Volleyball Spiking",
    "WalkingWithDog" : "Walking With Dog",
    "WallPushups" : "Wall Pushups",
    "WritingOnBoard" : "Writing On Board",
    "YoYo" : "Yo Yo",
}


def video_to_numpy(video_path, target_size=[720, 480], return_5d_array=False):
    """
    Reads a video file and converts it to a NumPy array.

    Args:
        video_path (str): Path to the video file.
        return_5d_array (bool): if true output will be a 5d array, else 4d
        target_size (list[int]): [w, h], defaults to [720, 480]

    Returns:
        frames (numpy.ndarray): Array of shape (B, N, H, W, C), where:
            B = batch
            N = number of frames
            H = frame height
            W = frame width
            C = number of color channels (3 for RGB)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()    # frame is [h, w, 3]
        if not ret:
            break  # Exit loop if no more frames

        frame = cv2.resize(frame, target_size[::-1], interpolation=cv2.INTER_CUBIC)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame_rgb)

    cap.release()

    frames = np.array(frames)
    frames = np.transpose(frames, (0,3,1,2))
    if return_5d_array == True:
        return np.expand_dims(frames, axis=0)
    else:
        return frames




class UCFDataSet(Dataset):

    def __init__(
        self,
        path : str=LOCAL_UCF_PATH,
        split : str="train",
        size : List[int] = [480,720],    # h, w
        ):

        self.folder_path = os.path.join(path, "UCF-101")
        self.split = split
        self.video_size = size

        if split == "train":
            video_list = "trainlist01.txt"
        elif split == "test":
            video_list = "testlist01.txt"
        else:
            raise ValueError (f"split must be 'train' or 'test', here get{split}")

        self.video_list_path = os.path.join(path, video_list)
        self.all_video_names = self._get_all_video_names()


    @property
    def _get_video_list(self) -> str:
        return self.video_list_path


    def _get_all_video_names(self) -> List[str]:

        with open(self.video_list_path, 'r', encoding='utf-8') as file:
            if self.split == "test":
                all_video_names = [line.strip() for line in file if line.strip()]
            elif self.split == "train":
                all_video_names = [line.strip().split(" ")[0] for line in file if line.strip()]
            else:
                raise ValueError (f"split must be 'train' or 'test', here get{self.split}")
        return all_video_names


    def __len__(self) -> int:
        return len(self.all_video_names)

    def _match_video_class_and_prompt(self, video_file:str) -> Tuple[str, str]:
        """
        video_file: str of video file name, e.g  'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c06.avi'

        return:
        """
        video_class = video_file.split("/")[0]
        prompt = UCF_LABELS[video_class]
        return prompt, video_class


    def __getitem__(self, idx:int) -> Tuple[str, np.ndarray]:

        video_file = self.all_video_names[idx]
        prompt, video_class = self._match_video_class_and_prompt(video_file)
        video_file = os.path.join(self.folder_path, video_file)
        video = video_to_numpy(video_file, target_size=self.video_size)

        # Skip this item if shape[0] < 100
        if video.shape[0] < 110:
            return self.__getitem__((idx + 1) % len(self))
        video = video[10:110, ...]
        return prompt, video

    def ucf_collate_fn(batch: List[Tuple[str, np.ndarray]]):
        """
        customized collate func to process a batch of samples

        Args:
            batch: List of samples, where each sample is (prompt, video)
            - prompt: str
            - video: np.ndarray of shape [100, 240, 320, 3]

        Returns:
            A tuple containing:
            - List[str]: List of prompts.
            - torch.Tensor: Batched videos of shape [batch_size, 100, 240, 320, 3].
        """
        #spearate prompts and videos
        prompts = [item[0] for item in batch]  # List of strings

        videos = [torch.from_numpy(item[1]) for item in batch]
        videos = torch.stack(videos, dim=0)

        return prompts, videos


if __name__ == "__main__":

    train_set = UCFDataSet(split='train')

    sample = train_set


    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=UCFDataSet.ucf_collate_fn, pin_memory=False)

    for idx, (prompts, videos) in enumerate(train_loader):
        if idx==5: break
        print("len(prompts): ", prompts)
        print("videos.shape: ", videos.shape)