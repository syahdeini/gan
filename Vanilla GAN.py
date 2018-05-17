
# coding: utf-8

# ## Learning Vanilla GAN by code
# 
# ### The theory
# GANs are deep neural networks scheme comprised of two deep nets, one is called the generator and the other is called the discriminator. First introduced in 2014 by Ian J. Goodfellow[1], the potential for GANs are huge. It can be used for image-to-image translations and general classifications. 
# 
# If you haven't read the paper, I really recommend you to skim it first.  
# The basic idea of GANs is to make the aforementioned neural networks (generator and discriminator) compete against each other. The process flow of GANs is shown below
# <img src="gan_diagram.png" width="300" height="200"></png>
# As an illustration, a single GAN can be seen as the combination of a counterfeiter (generator) and a cop (discriminator). The counterfeiter is trying to make false notes (money) to fool the cop. Meanwhile, the cop is also learning to detect them. Since both are training to beat each other, GANs have potential to do a lot of things better than conventional neural networks.
# 

# ### The maths
# 
# <img src="gan_formula.png" width="500" height="200"></img>
# 
# Basically gan is trying to learn the data distribution using the generative model with help of discriminative model.  
# 
# 
# The formula shown above simply mean that `we are trying to minimize the expected value of of` $D(x)$ `if we sample x from probablity distribution of data1` $P_{data}$ `, and maximize the value of` $G(z)$ `if we sample of z from distribution of noise` $P_{z}$.  
# 
# We are trying to optimum the value of this formula. 
# 
# <img src="gan_graph.png" width="500" height="200"> </img>
# 
# - the blue-dashed line is a discriminative distribution.  
# - black-dathed line is data distribution. $P_{x}$
# - green-solid line is from generative distribution $P_{g}$
# 
# This graph shows that first there are sampling from distribution z (vertical line from domain z to domain x). 
# As time goes (picture to right), model G will try to fit the distribution of data. At the end
# the generator fit the data and the the discriminator can't differentiate the distribution of data and generator. 
# At the end, they will reach a point at which both cannot improve because pg = pdata. 
# The discriminator is unable to differentiate between the two distributions, i.e. D(x) = 1 .
# 
# ### The algorithm
# Assume we have data distribution $P_{x}$.  
# We want to learn $P_{g}$ over data $x$. We define prior input noise $P_{z}$. First we take sample $z$ then we feed it into model $g(z,w)$ where w is the parameter to fit model *g* into the distribution $p_{x}$. 
# where $d(q,w)$ is a discriminator model that output single probability (logistic) if data comes from $P_{z}$ or $P_{x}$ (fake vs real)
# 
# The GAN algorithm can be seen
# <img src="gan_algo.png" height="600" width="550"></img>
# 
# 
# 

# ____
# ### Code implementation
# The code below show the implementation of GAN on tensorflow  
# source https://github.com/emsansone/GAN/blob/master/gan.py
# 
# GAMES is number of iteration all.   
# DISCR_UPDATE is when step to update discriminator  
# and GEN_UPDATE is when step to update generator

# In[1]:


import __future__ 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.animation as animation
import seaborn
from scipy.stats import norm
import os

GAMES = 10000
DISCR_UPDATE = 50
GEN_UPDATE = 1

tf.reset_default_graph()

class RealDistribution:
    def __init__(self):
        self.mu = 5
        self.sigma = 1

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        return samples


class NoiseDistribution:
    def __init__(self):
        self.low = 0
        self.high = 7

    def sample(self, N):
        samples = np.random.uniform(self.low, self.high, N)
        return samples


class GAN:
    # each scope DISC and GEN have different w and b parameter
    def linear(self, input, scope=None):
        init_w = tf.random_normal_initializer(stddev=0.1)
        init_b = tf.constant_initializer(0.0)
        # initializer only works for first time
        with tf.variable_scope(scope or 'linear'):              # USING SCOPE FOR FUTURE VERSION WITH MULTIPLE LAYERS
            w = tf.get_variable('w', [1,1], initializer=init_w)
            b = tf.get_variable('b', [1,1], initializer=init_b)
            return tf.add(tf.matmul(w, input), b)

    def generator(self, input):
        logits = self.linear(input, 'gen')
        return logits

    def discriminator(self, input):
        logits = self.linear(input, 'discr')
        pred = tf.sigmoid(logits)
        return pred

    def __init__(self):
        self.games = GAMES
        self.discriminator_steps = DISCR_UPDATE
        self.generator_steps = GEN_UPDATE
        self.learning_rate = 0.1
        self.num_samples = 10
        self.skip_log = 20
        
        self.noise = NoiseDistribution()
        self.data = RealDistribution()

        self.create_model()
        
    def create_model(self):
        # 1. Generator (G)
        with tf.variable_scope('GEN'):
            self.z = tf.placeholder(tf.float32, shape=(1, self.num_samples))
            # z to G. G(z)
            self.gen = self.generator(self.z)
            
        # 2. Discriminator (D)
        with tf.variable_scope('DISC') as scope:
            self.x = tf.placeholder(tf.float32, shape=(1, self.num_samples))
            self.discr_x = self.discriminator(self.x)
            scope.reuse_variables()
            self.discr_g_x = self.discriminator(self.gen)

        # 3. Losses
        self.loss_gen = tf.reduce_mean(tf.log(1-self.discr_g_x))
        self.loss_discr = tf.reduce_mean(-tf.log(self.discr_x) -tf.log(1-self.discr_g_x))

        # 4. Parameters
        self.gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        self.discr_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DISC')
        self.all_params = tf.trainable_variables()
        # 5. Optimizers (this optimizer who's actually updating the parameter)
        self.opt_gen = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.loss_gen,
            var_list=self.gen_params
        )
        self.opt_discr = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.loss_discr,
            var_list=self.discr_params
        )
        
        # 6. gradients (This function is just used to calculate the gradient after updating it 
        # using gradient descent)
        self.grad_discr = tf.gradients(self.loss_discr, self.discr_params)[0]
        self.grad_gen = tf.gradients(self.loss_gen, self.gen_params)[0]
        
    def train(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            x = self.data.sample(self.num_samples)
            objective_function = []
            grad_magn_discr = []
            grad_magn_gen = []
            eigs = []
            frames = 0
        
            for games in range(self.games):
                # Update discrimintator
                z = self.noise.sample(self.num_samples)
                for discr_steps in range(self.discriminator_steps):
                        loss_discr, _ = sess.run([self.loss_discr, self.opt_discr],{
                            self.x : np.reshape(x, (1, self.num_samples)),
                            self.z : np.reshape(z, (1,self.num_samples))
                        })
                grad_discr_val = sess.run(self.grad_discr, feed_dict={
                    self.x: np.reshape(x, (1, self.num_samples)),
                    self.z: np.reshape(z, (1, self.num_samples))
                })
                grad_magn_discr.append(np.linalg.norm(grad_discr_val))
                
                # intermediate visualization
                if games % self.skip_log == 0:
                    print('game %d: Loss: %.3f\tTarget loss: %.3f' %(games, -loss_discr, -2*np.log(2)))
                    self.intuition(sess, x)
                    frame = plt.gca()
                    frame.axes.get_yaxis().set_visible(False)
                    plt.draw()
                    plt.pause(0.01)
                    plt.clf()
                
                # update generator
                for gen_steps in range(self.generator_steps):
                    z = self.noise.sample(self.num_samples)
                    loss_gen, _ = sess.run([self.loss_gen, self.opt_gen], {
                        self.z: np.reshape(z, (1, self.num_samples))
                    })
                grad_gen_val = sess.run(self.grad_gen, feed_dict={
                    self.z: np.reshape(z,(1, self.num_samples))
                })
                grad_magn_gen.append(np.linalg.norm(grad_gen_val))
                
                
                # Intermediate visualization
                if games % self.skip_log == 0:
                    print('game %d: Loss: %.3f\tTarget loss: %.3f' % (games, -loss_discr, -2*np.log(2)))
                    self.intuition(sess, x)
                    frame = plt.gca()
                    frame.axes.get_yaxis().set_visible(False)
                    plt.savefig('img/img-'+str(games)+'.png')
                    plt.draw()
                    plt.pause(0.01)
                    plt.clf()
                    frames += 1

                objective_function.append(-loss_discr)

            ### below this code it's all for visualization    
            # Visualization
            plt.close()
            print('\nSaving summary...\n')
            gs = gridspec.GridSpec(2, 2)

            # Graphical interpretation
            plt.subplot(gs[0,0])
            self.intuition(sess, x)
            frame = plt.gca()
            frame.axes.get_yaxis().set_visible(False)

            # Objective function
            plt.subplot(gs[0,1])
            self.objective(objective_function, games)

            # Gradient discriminator
            plt.subplot(gs[1,0])
            plt.plot(range(self.games),grad_magn_discr)
            plt.title('Gradient magnitude - Discriminator')

            # Gradient generator
            plt.subplot(gs[1,1])
            plt.plot(range(self.games),grad_magn_gen)
            plt.title('Gradient magnitude - Generator')
            plt.savefig('img/summary_'+str(self.games)+'_'+str(self.discriminator_steps)+'_'+str(self.generator_steps)+'.eps')
            plt.savefig('img/summary_'+str(self.games)+'_'+str(self.discriminator_steps)+'_'+str(self.generator_steps)+'.png')

            # Animation
            print('\nCreating GIF animation...')
            fig = plt.figure()
            plt.axis('off')
            anim = animation.FuncAnimation(fig, self.animate, frames=frames)
            anim.save('img/img_'+str(self.games)+'_'+str(self.discriminator_steps)+'_'+str(self.generator_steps)+'.gif', writer='imagemagick', fps=int(120/self.skip_log))
            self.delete()

            
    def animate(self, i):
        print('Frame {}'.format(i))
        img = mpimg.imread('img/img-'+str(i*self.skip_log)+'.png')
        ax = plt.imshow(img)
        return ax

    def delete(self):
        i = 0
        while True:
            try:
                os.remove('img/img-'+str(i*self.skip_log)+'.png')
                i += 1
            except:
                return

    def intuition(self, sess, x):
        min_range = self.noise.low
        max_range = self.data.mu+2*self.data.sigma
        plt.xlim([min_range,max_range])
        plt.ylim([-0.6,1])

        # Lines
        plt.plot([min_range, max_range], [-0.5,-0.5], 'k-', lw=1)
        plt.plot([min_range, max_range], [0,0], 'k-', lw=1)

        # Samples
        num = 10
        z = self.noise.sample(num)
        plt.plot(z, -0.5*np.ones(num),'bo')
        out = sess.run(self.gen, {self.z: np.reshape(z, (1,self.num_samples))})
        plt.plot(np.transpose(out),                     np.zeros(num),'bo')

        # Arrows
        for i in range(num):
            plt.plot([z[i],out[0][i]],[-0.49,-0.01],'-k')

        # Real distribution
        x_range = np.linspace(min_range, max_range, 50)
        fit = norm.pdf(x_range, self.data.mu, self.data.sigma)
        plt.plot(x_range, fit, '-g')           

        # Real data
        plt.plot(x, np.zeros(self.num_samples),'go')

        # Discriminator
        num = 40*self.num_samples
        x_range = np.linspace(min_range, max_range, num)
        out = []
        for i in range(int(num/self.num_samples)):
            tmp = x_range[i*self.num_samples:(i+1)*self.num_samples]
            tmp = sess.run(self.discr_x, {self.x: np.reshape(tmp, (1,self.num_samples))})[0]
            for j in range(self.num_samples):
                out.append(tmp[j])
        plt.plot(x_range,                     np.array(out),'-b')

        plt.title('Graphical interpretation')

    def objective(self, objective, games):
        plt.plot(range(self.games),objective)
        plt.plot([1, games], [-2*np.log(2), -2*np.log(2)], 'r-', lw=1)
        plt.title('Objective vs. Iterations')


# In[8]:


model = GAN()
model.train()

