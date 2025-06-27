# ChessBotv2
A custom made neural network that plays chess

To be able to run this program you need an nvidea gpu capable of dynamic parallelism. It is also not supported on windows because of the assembly script.

This is (probably) the final version of a small project of mine. Put simply this project has the code to train and run a NN that plays chess, if you uncomment the "runInputRun" function call you can even play against it yourself, however the scope of the playing mechanism is fairly limited, given how this project's purpose was to test a custom reinforcment learning algorithm I cooked up. If you want to play against it, here's how the input/output works:
1-The bot only plays as whites, so it will always start first, it will output a source and destination, I hope what that means is self explainatory; This output will be of the
format x=something y=something, x represents the x coordinate(so the letters, where a=0 and h=7) and y the y coordinate, keep in mind the y coordinate is 0-indexed, 
so youll have to subtract 1 from the coordinate the real board shows.

2- After this you will be able to provide a move, This move is of the format "xSrc|ySrc xDst|yDst replacementFloat" I hope everything except the replacementFloat is self
explainatory, the replacementFloat indicates if the piece is going to change value (e.g when a pawn reaches the other end of the board) and the new value of the piece,
if replacementFloat is greater than 0, than no change to the piece type is made. Keep in mind the board is in the perspective of the white player.

Thats about it, the list of defined piece types is given by:
"""
//The pieces absolute values are defined as followed:
//pawn = 1; knight = 2; bishop = 3; rook = 4; queen = 5; king = 6;
//where the whites are positive and blacks are negative
"""

Obviously inputing a float that is not given will result in unpredicted behaviour.

I will now go over some of the technical details, and how you can manipulate some parameters if you want to try and find the optimal parameters yourself.

Every parameter with the exception of the layer sizes is defined as a macro, you can just change them and see what happens.
The layer sizes are defined in the main() function, again you can also just change this all you want, but do remember to retrain the model after the parameter changes

This model uses a normal feedforward neural network, the activation function in the hidden layers I use is leaky ReLU, and the final activation function is a softmax on the
upper 64 elements of the output vector and another on the lower 64 elements. It outputs a 128 element vector, the upper 64 elements represent the source position prob
distribution while the lower represent the dest position prob dist. Thats about it for the network itself, now the algorithm.

The general idea is fairly simple, I have a list of all boards the network has already classified a "best move" to, if the given board is in that list, I calculate
the gradient assuming the position the list shows as best and add it to an accumulator, if it isnt, I pick a random legal move and add the board to the list with the random selected move as the "best move" and do the same process I would otherwise. Since Im treating each game as its own batch, after each game I divide the accumulator vector by the total moves and do a gradient descent step with those values (multiplied by the learning step). This process is then done 20 times. This is because I need a model different enough from the original to be able to properly run it against the original. The different boards are reached by having the mutated model play against the original one.
After these 20 games, I will run 20 other games where I pitch the original model to this new mutated model, if the mutated model has a score above 0 (1 for a win, 0 for draw, 
-1 for loss), I then swap the old original model by this new mutated one, otherwise I simply discard the mutated model and repeat the process.
If you want more details feel free to look through the code.

All matrices are organized in col major to help with coalescense.



