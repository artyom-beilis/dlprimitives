import json
class NetConfig(object):
    def __init__(self):
        self.network=dict(
            inputs=[],
            outputs=[],
            operators=[],
        )

    @property
    def _operators(self):
        return self.network['operators']

    def save(self,path):
        with open(path,'w') as f:
            f.write(self.to_str())

    def set_outputs(self,outputs):
        self.network['outputs'] = outputs
        
    def add_input(self,name,shape,dtype='float'):
        self.network['inputs'].append(dict(
            name = name,
            shape=list(shape),
            dtype=dtype
        ))

    def add(self,op,name=None,inputs=None,outputs=None,options=None):
        if name is None:
            name = '%s_%d' % (op,len(self._operators))
        if inputs is None:
            assert len(self._operators) > 0 or len(self.network['inputs']) == 1,"You should specify input names if no operators added, or multiple inputs provided"
            if self._operators:
                inputs = self._operators[-1]["outputs"][:]
            else:
                inputs = [ self.network['inputs'][0]["name"] ]
        elif isinstance(inputs,(list,tuple)):
            inputs=list(inputs)
        else:
            inputs=[inputs]
        if outputs is None:
            outputs = [ name + "_output" ]
        elif isinstance(outputs,(list,tuple)):
            outputs = list(outputs)
        else:
            outputs = [outputs]
        if options is None:
            options = dict()

        self._operators.append(dict(type=op,name=name,inputs=inputs,outputs=outputs,options=options))

    def to_str(self):
        if not self.network['outputs'] and self._operators:
            n=dict(inputs = self.network['inputs'],operators=self._operators,outputs=self._operators[-1]['outputs'])
        else:
            n=self.network
        return json.dumps(n,indent=2)

    def __str__(self):
        return self.to_str()
        
