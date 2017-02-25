import numpy as np
import matplotlib.pyplot as plt

# Increase this if you want more than 100 data points
DATASET_SIZE = 100

# If our regression works our tuple should be close to these values
ACTUAL_SLOPE = .04
ACTUAL_INTERCEPT = 4.5
NOISE = 1  # Manipulate the noise of the data points

# x = (0,1,2, ... n)
x = np.arange(DATASET_SIZE)
y = np.random.uniform(ACTUAL_INTERCEPT - NOISE , ACTUAL_INTERCEPT + NOISE, DATASET_SIZE)

#Bias the data so it has the desired slope
for i in range(DATASET_SIZE):
	y[i] += ACTUAL_SLOPE*i

plt.plot(x,y,'d')

axes = plt.gca() 
axes.set_ylim(0,10) # Zooms the y axis out a bit for a better view

# returns the sum of squares of difference between set of actual values
# and a set of predicted values
def cost_function(y, predicted_values):
	total_cost = 0
	for i in range(DATASET_SIZE):
		total_cost += (y[i]-predicted_values[i])**2
	return total_cost

def get_prediction(m,b):
	predicted_values = []
	for i in range(DATASET_SIZE):
		predicted_values.append(m*i+b)
	return predicted_values

# Walk right, left, up or down n times, descending to
# whatever (m,b) yields the lowest cost
def gradient_descent(m,b,y,predicted_values,n):
	if n == 0:
		return (m,b)
	predicted_values_up = get_prediction(m+.001, b)
	predicted_values_right = get_prediction(m, b+.01)
	predicted_values_left = get_prediction(m, b-.01)
	predicted_values_down = get_prediction(m-.001, b)

	cost_up = cost_function(y, predicted_values_up)
	cost_right = cost_function(y, predicted_values_right)
	cost_left = cost_function(y, predicted_values_left)
	cost_down = cost_function(y, predicted_values_down)
	min_cost = min(cost_up, cost_right, cost_left, cost_down)
	if cost_up == min_cost:
		return gradient_descent(m+.001, b, y, predicted_values, n-1)
	if cost_right == min_cost:
		return gradient_descent(m, b+.01, y, predicted_values, n-1)
	if cost_left == min_cost:
		return gradient_descent(m, b-.01, y, predicted_values, n-1)
	if cost_down == min_cost:
		return gradient_descent(m-.001, b, y, predicted_values, n-1)

# Get and show baseline prediction to see if gradient descent
# faired much better
base_line_prediction = get_prediction(.03, 3.5)
plt.plot(x,base_line_prediction)


tupl = gradient_descent(.03,3.5,y,[],300)
m = tupl[0]
b = tupl[1]
predicted_values = get_prediction(m,b)

plt.plot(x,predicted_values)
print tupl
plt.show()	# all done! :)		
