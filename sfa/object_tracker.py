from motion_vectors import MotionVector
import math


class ObjectTracker:

    def __init__(self):
        self.timestamp = 0
        self.next_id = 0
        self.first_frame = True
        self.detected_objects = []
        # self.detected_objects_next_state = []
        self.max_distance = 3.15
        self.objects_identified = []
        self.all_detections = []

    def identify_object(self, detections, t, delta_t):
        self.objects_identified = []

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

                        prev_x = obj['Location'].x
                        prev_y = obj['Location'].y
                        prev_h = obj['Heading']
                        new_x = object_mv['Location'].x
                        new_y = object_mv['Location'].y
                        new_h = object_mv['Heading']

                        x = new_x - prev_x
                        y = new_y - prev_y
                        r = (x**2 + y**2)**(1/2)

                        h = new_h - prev_h

                        # use kalman filter to improve estimation (?)
                        if r <= self.max_distance and obj['Classification'] == object_mv['Classification']:
                            speed = r/delta_t  #speed relative to the car
                            acceleration = (speed - obj['Speed'])/delta_t
                            yaw_rate = h/delta_t
                            obj['Location'].x = object_mv['Location'].x
                            obj['Location'].y = object_mv['Location'].y
                            obj['Location'].z = object_mv['Location'].z
                            obj['Speed'] = speed
                            obj['Acceleration'] = acceleration
                            obj['Yaw_rate'] = yaw_rate
                            obj['Heading'] = object_mv['Heading']
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
        # self.predict_next_state(self.detected_objects, delta_t)

    # def predict_next_state(self, objects, delta_t):
    #     self.detected_objects_next_state = []

    #     for object in objects:
    #         prev_vel = object['Speed']
    #         prev_accel = object['Acceleration']
    #         prev_yaw_rate = object['Yaw_rate']
    #         prev_heading = object['Heading']

    #         delta_x = prev_vel*math.sin(prev_heading)*delta_t
    #         delta_y = prev_vel*math.cos(prev_heading)*delta_t
    #         if delta_t > 0:
    #             delta_v = ((delta_x**2 + delta_y**2)**(1/2))/delta_t
    #             object['Speed'] = delta_v

    #         object['Location'].x += delta_x
    #         object['Location'].y += delta_y

    #         self.detected_objects_next_state.append(object)

    def show_objects(self):
        for obj in self.detected_objects:
            print("Id:" + str(obj['Id']))
            print("Location:" + str(obj['Location'].x) + " " +
                  str(obj['Location'].y) + " " + str(obj['Location'].z) + " (m)")
            print("Speed:" + str(obj['Speed']) + " (m/s)")
            print("Acceleration:" + str(obj['Acceleration']) + " (m/s²)")
            print("Yaw_rate:" + str(obj['Yaw_rate']) + " (rad/s)")
            print("Heading:" + str(obj['Heading']) + " (rad)")
            print("Dimension:" + str(obj['Dimension'].x) + " " +
                  str(obj['Dimension'].y) + " " + str(obj['Dimension'].z) + " (m)")
            print("Classification:" + str(obj['Classification']))
            print("Class_confidence:" + str(obj['Class_confidence']))
            print("Timestamp:" + str(obj['Timestamp']) + " (s)")
            print("\n")
