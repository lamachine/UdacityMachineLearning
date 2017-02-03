from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    #initialize error to zero
    totalError = 0
    #for every row in points, using for loop
    for i in range(0,len(points)):
        #get the x value
        x = points[i,0]
        #get the y value
        y = points[i,1]
        #get the difference, square it, and add it to the total
        totalError += (y - (m * x + b)) **2
    #get the average
    return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #starting b and m values
    b = starting_b
    m = starting_m

    #gradient descent
    for i in range(num_iterations):
        #update b and m with new more accurate b and m by performing
        #this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

#this is the magic
def step_gradient(b_current, m_current, points, learning_rate):
    #starting points for our gradients
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #direction with respect to b and m
        #compute partial derivatives of our error function
        b_gradient += -(2/N) *     (y - ((m_current * x) + b_current))
        m_gradient +=  (2/N) * x * (y - ((m_current * x) + b_current))

    #update our b and m values using out partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def run():
    #Step 1 - collect our data
    points = genfromtxt('data.csv', delimiter=',' )
    print (points)
    #Step 2 - define our hyperparamters
    #how fast should our model converge
    learning_rate = 0.0001
    #y = mx + b
    initial_b = 0 #y intercept guess
    initial_m = 0 #slope guess
    num_iterations = 1
    #Step 3 - train our model
    print ('Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ('Ending gradient descent at b = {1}, m = {2}, error = {3}'.format(num_iterations,b, m,compute_error_for_line_given_points(b,m, points)))
    

if __name__ == '__main__' :
    run()
