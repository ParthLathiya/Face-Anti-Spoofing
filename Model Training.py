from ultralytics import YOLO

model = YOLO('new.pt')
def main():
    model.train(data='Face Anti-Spoofing(mini project)\\HONORS OEP\\Data Split\\data.yaml', epochs=5)

if __name__ == '__main__':
    main( )