# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import flwr as fl
import socket

if __name__ == "__main__":
    ip_address = '127.0.0.1'
    server_address=ip_address + ':8080'


    print(ip_address)
    
    fl.server.start_server(server_address=server_address,config={"num_rounds": 5})
    #fl.server.start_server(config={"num_rounds": 10})
    
    
    
    

    
    

#/home/sjx/anaconda3/envs/pytorch/lib/python3.6/site-packages/
#/home/sjx/anaconda3/envs/pytorch-gpu/lib/python3.7/site-packages/flwr/server/strategy