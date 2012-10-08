pyemergent
==========

Specify, run, and analyze emergent experiments in Python.

Emergent is a neural network simulator.
http://grey.colorado.edu/emergent/index.php/Main_Page

Disclaimer
==========

This was mainly used for my own simulations. While I think this has
the potential to be useful to others, I think it would require more
work.

Mini-how-to
===========

The main action is in emergent.py which has the base class one would inherit from to define a new experiment. That could look as follows:

```python
@pools.register('my_experiment')
class MyExperiment(emergent.Base):
   def __init__(self):
       self.proj_name = 'my_proj' # emergent proj file
       self.prefix = '/home/wiecki/emergent' # directory
       self.tags = ['intact', 'lesioned']

       self.flag['tag'] = 'intact'
       self.flag['lesioned'] = False
       self.flags.append(self.flag)

       self.flag['tag'] = 'lesioned'
       self.flag['lesioned'] = True
       self.flags.append(self.flag)

   def analyze(self):
       for tag in self.tags:
           pylab.plot(self.data[tag]['minus_cycles'])
```

You could then run this model as follows:
```bash
mpirun -n 40 python emergent.py --run --mpi --batches 20 --group my_experiment
```
This would then launch 40 emergent processes (20 batches * 2 conditions (intact and lesioned)).

If you call it afterwards with --analyze set, it would aggregate all log-files into one numpy array and call the analyze() method defined above. I also recoded the groupby function of emergent. However, only later I learned about a package called pandas which does this and much more, much better. So in the future I might change the data structure to be a pandas dataframe instead of a numpy array which gives it more flexibility.


License
=======

Copyright (c) 2012, Thomas Wiecki
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.