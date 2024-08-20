###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
"""
Defines helper utilities to work with dlprimitives C++ library

All its classes imported directly to top level dlprim module

"""

import json


class NetConfig(object):
    """
    Helper class that allows generation of network definitions in JSON format
    """
    def __init__(self):
        """
        Create an empty network
        """
        self.network=dict(
            inputs=[],
            outputs=[],
            operators=[],
        )

    @property
    def _operators(self):
        """
        Get list of all operators added to the network
        """
        return self.network['operators']

    def save(self,path):
        """
        Save network definition to file named path
        """
        with open(path,'w') as f:
            f.write(self.to_str())

    def set_outputs(self,outputs):
        """
        Specify network outputs, outputs need to be list of strings that define output tensor names
        for example ["prob","loss"]

        Note: if no output is set the output of the last operator is considered network output
        """
        self.network['outputs'] = outputs
        
    def add_input(self,name,shape,dtype='float'):
        """
        Define network input tensor

        Parameters:
        name: string name of input tensor
        shape: list/tuple of integers shape of input tensor
        dtype: type of input tensor: float, int32, etc.
        """
        self.network['inputs'].append(dict(
            name = name,
            shape=list(shape),
            dtype=dtype
        ))

    def add(self,op,name=None,inputs=None,outputs=None,options=None):
        """
        Add new operator to network, it is appended to the end.
        Parameters:

        op: string operator type, for example Activation
        name: string a unique  name of the operator, if not provided it is auto-generated
        inputs: string or list of strings operator inputs, if not provided it takes outputs of the provious operator or network input
          in no operator exits (must be single input)
        outputs: string or list of strings, if not defined single output name is autogeneratord
        options: dict - dictionart of options for operator op.
        """
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

    def to_json(self):
        """
        Return json representaion of the network, note it is python dictionary that can be serialized to json
        """
        if not self.network['outputs'] and self._operators:
            n=dict(inputs = self.network['inputs'],operators=self._operators,outputs=self._operators[-1]['outputs'])
        else:
            n=self.network
        return n

    def to_str(self):
        """
        Create json representation of the network as string
        """
        return json.dumps(self.to_json(),indent=2)

    def __str__(self):
        """
        returns self.to_str()
        """
        return self.to_str()
        
