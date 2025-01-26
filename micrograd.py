import random
import math


class Value: 
    """
    Represents a value in a computational graph, typically used for building an autograd system.
    It stores the value of a computation and tracks its relationship with other values for gradient computation (backpropagation).
    """
    def __init__(self, data, _children=(), _op='', label=''):
        # A constructor for the class to create object 
        # It's called automatically when creating a new instance of the class
        # self is similar to this in Java, but it has to be specified in Python and must come as the first parameter
        # :_children=(): an optional parameter with a default value of an empty tuple ().
        # _ before _children and _op is a convention in Python indicating that these attributes are intended to be private to the class.
        # It suggests that they are internal details, not meant to be accessed directly outside the class, though it does not prevent access.
        self.data = data
        self.grad = 0.0              # gradient = 0 meaning it does not affect the loss function value as default
                                     # it represent the derivatives of L (loss function) in regards to each node/weight of the model in backpropagation
        self._backward = lambda: None
        self._prev = set(_children)  # keep track of the "previous" Value objects involved in the operation
                                     # It's a set of other Value objects that were used in the current opeartion, enabling backtracking during gradient computation
        self._op = _op               # Stores the operation performed to create the current Value object.
                                     # This helps in determining how to compute derivatives during backpropagation or other opeartions.
        self.label = label
        
    def __repr__(self):
        # The representation method defining how the object is represented as a string when printed or logged
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad    # += to accumulate the gradients
            other.grad += 1.0 * out.grad
        out._backward = _backward # the _backward function as a property (attribute) of the Value object out
                                  # Later, when the _backward() method is called, the _backward function of each node
                                  # node is invoked to propagate gradients to its input recursively.
        
        return out

    def __mul__(self, other):
        # check if `other` is a Value object, if not wrap it in a Value object before doing op
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, other): 
        # Handles the case when 'other * self' is called.
        # This function is called when the multiplication operation is attempted and 'self' 
        # is on the righ side of the '*' operator, e.g 3 * Value(2)

        # Python first tries to call the left operand's __mul__ method.
        # If the left operand doesn't have __mul__ or it doesn't know how to handle
        # multipliction with 'self', Python calls the right operand's __rmul__ method.
        
        # Step 1: Python tries 3.__mul__(Value(2.0)), which fails because int doesn’t know how to handle Value.
        # Step 2: Python falls back to Value(2.0).__rmul__(3).
        # Step 3: Inside __rmul__, self is Value(2.0), and other is 3
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other): # self / other
        # Instead of implementing division logic directly, this method leverages the __mul__ method
        # by multiplying `self` with the reciprocal of `other` (computed as other**(-1).
        # This ensures consisten behavior across multiplication and division, 
        # while keeping the implemetation simple and centralized.
        return self * other**(-1)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)
        
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __rtruediv__(self, other): # other / self
        return other * self**(-1)



# In[41]:


class Neuron:

    def __init__(self, nin):
        # `self.w` is a list of `Value` objects with the length of `nin`, representing the weights of the neuron.
        # Each weight is initialized to a random value between -1 and 1.
        # random.uniform() provides a uniform distribution, meaning every number in the range has an equal probability of being selected.
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # This is called a list comprehension in Python

        # `self.b` is a single `Value` object representing the bias of the neuron.
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # This method is invoked when an instance of the class is called like a function (e.g, `n(x)`)
        # print(list(zip(self.w, x))) # [(Value(data=0.39262953523948485), 2.0), (Value(data=-0.5394147081327125), 3.0)]
        # 1. Pair up each weight with its corresponding input value using `zip(self.w, x)`
        # For example, self.w = [w1, w2] and x = [x1, x2] --> zip(self.w, x) = [(w1, x1), (w2, x2)]
        wx_pair = zip(self.w, x)

        # 2. Multiply each weight and its corresponding input using a generator expression
        # This creates a sequence of products: [w1*x1, w2*x2, ...]
        weighted_inputs = (wi * xi for wi, xi in wx_pair)

        # 3. Sum up all the weighted inputs and add the bias `self.b`
        # Thge `sum()` function calculates the sum of the weighted inputs,
        # and the bias term `self.b` is added as the starting value.
        # The result `act` is the total activation of the neuron before applying an activation function.
        act = sum(weighted_inputs, self.b)
        out = act.tanh()
        
        return out

    def parameters(self):
        # Return all the parameters of a single neuron, which are `self.w` and `self.b` as lists
        return self.w + [self.b]

    def __repr__(self):
        return f"'Linear' Neuron({len(self.w)})"


class Layer:

    def __init__(self, nin, nout):
        # Create a list of neurons. The layer contains `nout` neurons. Each neuron takes `nin` inputs.
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # Process the input `x` through each neuron in the layer.
        # For every neuron in the layer: 
        #  - pass the input `x` to the neuron's __call__ method.
        #  - Collect the output from each neuron in a list
        outs = [n(x) for n in self.neurons]
        
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []

        # Iterate over all neurons in the layer
        # Call the parameters method of each Neuron to gets its weights and bias.
        # Combine all the paraneter into a single list using extend
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        
        return params

        # List comprehension logic
        # return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


    
class MLP:

    def __init__(self, nin, nouts):
        # Define the sizes of all layers in the MLP.
        # - Combine the input size `nin` with the list of output sizes `nouts`.
        # - Example: If nin=3 and nouts=[4, 5, 2], then sz = [3, 4, 5, 2]
        sz = [nin] + nouts

        # Create the layers of the MLP.
        #  - Iterate throught `sz` list to define each layer.
        #  - Each layer connects sz[i] inputs to sz[i+1] outputs.
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        # Pass the input `x` through each layer in the MLP sequentially.
        for layer in self.layers:
            # Update `x` by feeding it through the current lyare
            x = layer(x)

        return x

    def parameters(self):
        # Iterate over all layers in the MLP, and call the parameter method of each Layer to retrive its parameters
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

