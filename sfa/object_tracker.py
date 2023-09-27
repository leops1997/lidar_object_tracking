from motion_vectors import MotionVector

class ObjectTracker:

    def __init__(self):
        self.timestamp = 0
        self.next_id = 1
        self.first_frame = True
        self.detected_objects = []
        self.max_distance = 3.15
        self.objects_identified = []
        self.all_detections = []

    def identify_object(self, detections, t, delta_t):
        self.objects_identified =[]

        if len(detections) > 0:
            for det in detections:
                object_identified = False
                object_mv = MotionVector().transform_to_motion_vector(det)
                if self.first_frame:
                    object_mv['Id'] = self.next_id
                    object_mv['Timestamp'] = t
                    self.next_id += 1
                    self.detected_objects.append(object_mv)
                    self.all_detections.append(object_mv)
                else:
                    for obj in self.detected_objects:
                        
                        prev_x = obj['Position'].x
                        prev_y = obj['Position'].y
                        new_x = object_mv['Position'].x
                        new_y = object_mv['Position'].y

                        x = new_x - prev_x
                        y = new_y - prev_y
                        r = (x**2 + y**2)**(1/2)

                        if r <= self.max_distance and obj['Classification'] == object_mv['Classification']:
                            velocity = r/delta_t
                            acceleration = velocity/delta_t
                            obj['Position'].x = object_mv['Position'].x
                            obj['Position'].y = object_mv['Position'].y
                            obj['Position'].z = object_mv['Position'].z
                            obj['Velocity'] = velocity
                            obj['Acceleration'] = acceleration
                            obj['Yaw_rate'] = object_mv['Yaw_rate']
                            obj['Heading'] = object_mv['Heading']
                            obj['Angle'] = object_mv['Angle']
                            obj['Dimension'].x = object_mv['Dimension'].x
                            obj['Dimension'].y = object_mv['Dimension'].y
                            obj['Dimension'].z = object_mv['Dimension'].z
                            obj['Classification'] = object_mv['Classification']
                            obj['Class_confidence'] = object_mv['Class_confidence']
                            obj['Timestamp'] = t

                            object_identified = True

                            self.objects_identified.append(obj)
                            self.all_detections.append(obj)
                            break

                if not object_identified:
                    object_mv['Id'] = self.next_id
                    object_mv['Timestamp'] = t
                    self.next_id += 1
                    self.objects_identified.append(object_mv)
                    self.all_detections.append(object_mv)
            self.detected_objects = self.objects_identified
        self.first_frame = False

    def show_objects(self):
        for obj in self.detected_objects:
            print("Position:" + str(obj['Position'].x) + " " +
                  str(obj['Position'].y) + " " + str(obj['Position'].z))
            print("Velocity:" + str(obj['Velocity']))
            print("Acceleration:" + str(obj['Acceleration']))
            print("Yaw_rate:" + str(obj['Yaw_rate']))
            print("Heading:" + str(obj['Heading']))
            print("Angle:" + str(obj['Angle']))
            print("Dimension:" + str(obj['Dimension'].x) + " " +
                  str(obj['Dimension'].y) + " " + str(obj['Dimension'].z))
            print("Classification:" + str(obj['Classification']))
            print("Id:" + str(obj['Id']))
            print("Class_confidence:" + str(obj['Class_confidence']))
            print("Timestamp:" + str(obj['Timestamp']))
            print("\n")

