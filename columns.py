y_column = ['price']

X_columns = ["rating", 
             "has_5g", 
             "has_nfc", 
             "has_ir_blaster", 
             "processor_speed", 
             "battery_capacity", 
             "fast_charging", 
             "ram_capacity", 
             "internal_memory", 
             "refresh_rate", 
             "num_rear_cameras", 
             "brand_name_freq_encoded", 
             "processor_name_freq_encoded", 
             "processor_brand_freq_encoded", 
             "num_cores_freq_encoded", 
             "resolution_freq_encoded", 
             "num_front_cameras_freq_encoded", 
             "os_freq_encoded", 
             "primary_camera_rear_freq_encoded", 
             "primary_camera_front_freq_encoded", 
             "extended_memory_freq_encoded"]

outlier_columns = ["battery_capacity", 
                   "ram_capacity", 
                   "internal_memory"]

scaling_columns = ["rating", 
                   "processor_speed", 
                   "battery_capacity",
                   "fast_charging",
                   "ram_capacity", 
                   "internal_memory", 
                   "refresh_rate", 
                   "num_rear_cameras"]


cat_columns = ['brand_name',
               'processor_name', 
               'processor_brand', 
               'num_cores', 
               'resolution', 
               'num_front_cameras', 
               'os', 
               'primary_camera_rear', 
               'primary_camera_front', 
               'extended_memory']
               