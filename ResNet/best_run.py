from resnet_run import resnet_run

def best_run():
    resnet_run(starting_model = "best_model.pth")

if __name__ == '__main__':
    best_run()
    print("Run finished, please find results in ResNet/results")