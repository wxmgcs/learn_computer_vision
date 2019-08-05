import sys
import getopt
def usage():
    print ("usage")

def main(argv):
    grammar = "kant.xml"
    try:
        opts, args = getopt.getopt(argv, "hg:d", ["help", "grammar="])
        print ("opts:",opts)
        print ("args:",args)
    except getopt.GetoptError:
        usage()
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])