from retrieval import Retrieval
import cv2
from os.path import dirname, abspath, join



# Example of usage across the Building phase, opencv inference RT phase and testing phase

def main():
    
    retr = Retrieval('m_blacklist.pt', debug=True)
    
    # building embedding db
    #retr.buildBlacklistEmbeddings()
    
    # inference
    
    vid = cv2.VideoCapture(0)
    while True:
        _, frame = vid.read()
        r = retr.evaluateFrame(frame)
        if(r):
            print("Found")
        else:
            print("Not found")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # testing
    
    # retr.computeAccuracy( join(dirname(abspath(__file__)), 'datasets','lfw' ), join(dirname(abspath(__file__)), 'datasets','TP' ))
    

if __name__ == "__main__":
    main()