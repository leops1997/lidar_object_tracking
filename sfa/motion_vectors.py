class Pose3D:
    x = 0.0
    y = 0.0
    z = 0.0


class MotionVector:

    def __init__(self):
        self.position = Pose3D()
        self.velocity = 0.0
        self.acceleration = 0.0
        self.yaw_rate = 0.0
        self.heading = 0.0
        self.angle = 0.0
        self.dimension = Pose3D()
        self.classification = 0
        self.id = 0
        self.class_confidence = 0
        self.timestamp = 0

    def transform_to_motion_vector(self, detection):

        self.classification, self.position.x, self.position.y, self.position.z, self.dimension.x, self.dimension.y, self.dimension.z, self.heading = detection

        mv = {"Position": self.position,
                "Velocity": self.velocity,
                "Acceleration": self.acceleration,
                "Yaw_rate": self.yaw_rate,
                "Heading": self.heading,
                "Angle": self.angle,
                "Dimension": self.dimension,
                "Classification": self.classification,
                "Id": self.id,
                "Class_confidence": self.class_confidence,
                "Timestamp": self.timestamp}

        return mv
