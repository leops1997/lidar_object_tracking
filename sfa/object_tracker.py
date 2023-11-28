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

    def identify_object(self, detections, t, delta_t, id=None, have_id=False):
        self.objects_identified = []

        if len(detections) > 0:
            for i,det in enumerate(detections):
                object_identified = False
                object_mv = MotionVector().transform_to_motion_vector(det)
                if have_id:
                    if i <= len(id)-1:
                        object_mv['Id'] = id[i]
                        self.next_id = id[i]+1
                        object_mv['Timestamp'] = t
                    else:
                        object_mv['Id'] = self.next_id
                        self.next_id += 1
                        object_mv['Timestamp'] = t
                    self.objects_identified.append(object_mv)
                    self.all_detections.append(object_mv)
                else:
                    if self.first_frame:
                        object_mv['Id'] = self.next_id
                        self.next_id += 1
                        object_mv['Timestamp'] = t
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
                                obj['x'] = obj['Location'].x
                                obj['y'] = obj['Location'].y
                                obj['z'] = obj['Location'].z
                                obj['w'] = obj['Dimension'].y
                                obj['l'] = obj['Dimension'].x
                                obj['h'] = obj['Dimension'].z
                                self.all_detections.append(obj)
                                break

                    if not object_identified:
                        object_mv['Id'] = self.next_id
                        object_mv['Timestamp'] = t
                        self.next_id += 1
                        self.objects_identified.append(object_mv)
                        object_mv['x'] = object_mv['Location'].x
                        object_mv['y'] = object_mv['Location'].y
                        object_mv['z'] = object_mv['Location'].z
                        object_mv['w'] = object_mv['Dimension'].y
                        object_mv['l'] = object_mv['Dimension'].x
                        object_mv['h'] = object_mv['Dimension'].z
                        self.all_detections.append(object_mv)
            self.detected_objects = self.objects_identified
        self.first_frame = False

    def show_objects(self, sensor):
        for obj in self.detected_objects:
            print("Sensor name: " + sensor)
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
