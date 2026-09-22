class SimpleArgs:
    def __init__(
        self,
        pretrained=None,
        num_perception_frame=1,
        in_height=256,
        in_width=256,
    ):
        self.pretrained = pretrained
        self.num_perception_frame = num_perception_frame
        self.in_height = in_height
        self.in_width = in_width
