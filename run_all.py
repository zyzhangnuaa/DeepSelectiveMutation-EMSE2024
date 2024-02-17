import subprocess

subprocess.run(["python", "prediction_Label_Diversity_Random.py",
                    "-subject_name", "lenet5",
                    "-dataset", "mnist"])

subprocess.run(["python", "prediction_Label_Diversity_Random.py",
                    "-subject_name", "mnist",
                    "-dataset", "mnist"])

subprocess.run(["python", "prediction_Label_Diversity_Random.py",
                    "-subject_name", "svhn",
                    "-dataset", "svhn"])

# subprocess.run(["python", "prediction_Label_Diversity_Random.py",
#                     "-subject_name", "cifar10",
#                     "-dataset", "cifar10"])