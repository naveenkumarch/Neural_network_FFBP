from FFNBClass import Neuron_Network
class NeuralNetHolder:

    def __init__(self):
        """ Initializing the neural network with previously obtained architecture and replacing the weight vectors with previously trained weights """
        super().__init__()
        No_of_Inputs = 2
        No_of_Outputs = 2
        No_of_Hidden_nodes = 12
        """ Defining the min and max values of the data used in training for normalizing the input values sent by game and denormalize the predicted output values from neural network """  
        self.max_X_Distance =  636.2896274868181
        self.min_X_Distance = -596.6699167751983
        self.max_Y_Distance =  743.1598623474754
        self.min_Y_Distance =  65.6244797954829
        self.max_X_Velocity =  7.994655683732954
        self.min_X_Velocity =  -6.139791884122744
        self.max_Y_Velocity =  6.116163006403942
        self.min_Y_Velocity =  -5.779221733928611
        self.Neural_Net = Neuron_Network(No_of_Inputs, No_of_Hidden_nodes, No_of_Outputs)
        self.Neural_Net.Hidden_Layer_Weights = [[-2.4968545286197834, -1.8753602229555426, -0.212544244291629], [-1.7630022249709958, -3.6728753504716702, 0.9029212995412115], [-9.92308792895824, 18.605900320220044, 0.6546005968930644], [-2.4482999114771995, -1.517816946765758, -0.9193463164391101], [-2.3427861053090684, -2.4881000020941877, 0.4629152770160724], [-2.1591465483332413, 1.0195709398508257, -3.550975138336682], [-4.121604475036676, 1.2541841992381966, 0.20872225266025077], [-2.794714837157948, -0.6250218903568433, -0.9508382423169754], [-2.171501881731379, -2.860403977932674, 0.45023268515928966], [-7.574606539172206, 5.796893890015888, 0.8325562788065618], [-2.3949093030515787, -1.6691739704587119, -0.8994153916849774], [-2.5057827237537236, -1.833523946060227, -0.15265344756899354]]
        self.Neural_Net.Output_Layer_Weights =  [[0.5339576155454724, -7.163855899626589, 4.441573522337238, -0.8487519667092871, 0.194328665944557, -6.253588662045125, 10.355395474689958, -0.5546973711452573, 1.3109277184619805, -2.8628613991153036, -3.4019242278486903, 0.920569758736398, -9.436494568306678], [-1.2778954480096152, 0.7155347068753504, 1.642050336134636, 1.847449069077208, 0.6888835859247565, 1.1005203424912922, 1.8925919549669181, -0.6795836727331039, 0.41572054666867386, 1.2533245105144883, -3.297414893260861, 0.7326422000597372, 0.6620055115639853]]
    def predict(self, input_row):
        i = 0
        print("INput", input_row)
        for i in range(len(input_row)):
            if input_row[i] == ',':
                x_distance = float(input_row[0:i-1])
                y_distance = float(input_row[i+1:])
            else:
                pass
        """ The input values string read from game is converted to float variables for passing on to neural network """
        x_conv = (x_distance - self.min_X_Distance)/(self.max_X_Distance - self.min_X_Distance)
        y_conv = (y_distance - self.min_Y_Distance)/(self.max_Y_Distance - self.min_Y_Distance)
        current_pos = []
        current_pos.append(x_conv)
        current_pos.append(y_conv)
        Output_pred = self.Neural_Net.Predict(current_pos)       # passing on the converted values to the neural netowrk predict function
        Output_pred[0] = (Output_pred[0]*(self.max_X_Velocity - self.min_X_Velocity))+self.min_X_Velocity
        Output_pred[1] = (Output_pred[1]*(self.max_Y_Velocity - self.min_Y_Velocity))+self.min_Y_Velocity
        """ denormalize the output prediction values sent by neural net before passing them onto the game """
        return Output_pred
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity


