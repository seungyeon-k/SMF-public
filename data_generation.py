import yaml

from functions.data_generation import generate_data

if __name__ == '__main__':

    # set parameters
    num_objects = 500
    num_pnts = 512
    num_rots = 1
    visualize_object = False
    visualize_pc = False

    # data configuration
    config_total = {}
    dir_name = 'datasets/example'

    for shape in ["box", "ellipsoid", "cone"]:

        # shape parameters
        if shape is "box":
            e1 = {"random": False, "value": 0.0}
            e2 = {"random": False, "value": 0.0}
            k = {"random": False, "value": 0.0}
        elif shape is "ellipsoid":
            e1 = {"random": False, "value": 1.0}
            e2 = {"random": False, "value": 1.0}
            k = {"random": False, "value": 0.0}
        elif shape is "cylinder":
            e1 = {"random": False, "value": 0.0}
            e2 = {"random": False, "value": 1.0}
            k = {"random": False, "value": 0.0}   
        elif shape is "cone":
            e1 = {"random": False, "value": 0.0}
            e2 = {"random": False, "value": 1.0}
            k = {"random": False, "value": -1.0}                      

        for fit in ['tall', 'normal', 'short']:
            
            # size parameters
            if fit == 'tall':
                a1 = {"random": False, "value": 1}
                a2 = {"random": True, "min": 0.33, "max": 3}
                a3 = {"random": True, "min": 3, "max": 8}
            elif fit == 'normal':
                a1 = {"random": False, "value": 1}
                a2 = {"random": True, "min": 0.33, "max": 3}
                a3 = {"random": True, "min": 0.33, "max": 3}
            elif fit == 'short':
                a1 = {"random": False, "value": 1}
                a2 = {"random": True, "min": 0.33, "max": 3}
                a3 = {"random": True, "min": 0.125, "max": 0.33}
            else:
                ValueError('invalid fit text')

            config_total[f"{shape}_{fit}"] = {
                "dir_name": dir_name,
                f"{shape}": {
                    "a1": a1,
                    "a2": a2,
                    "a3": a3,
                    "sort": False,
                    "e1": e1,
                    "e2": e2,
                    "k": k,
                }
            }
    
    for name, config_ in config_total.items():
        
        # initialize
        dir_name = config_["dir_name"]
        del config_["dir_name"]
        config = config_

        # generate data
        print(name)
        generate_data(
            config, 
            num_objects=num_objects,  
            num_pnts=num_pnts,  
            num_rots=num_rots,
            append=True, 
            visualize_object=visualize_object,
            visualize_pc=visualize_pc, 
            dir_name=dir_name,
            dir_key=name
        )

        with open(dir_name + '/' + name + '/config.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)