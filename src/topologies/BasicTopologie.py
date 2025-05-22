from mininet.topo import Topo
from mininet.node import OVSKernelSwitch

class SimulationTopo(Topo):
    def __init__(self, queue_size=100, **opts):
        self.queue_size = queue_size
        super(SimulationTopo, self).__init__(**opts)

    def build(self):
        # Hosts
        s1 = self.addHost('s1', ip='10.0.0.1/24')
        s2 = self.addHost('s2', ip='10.0.0.2/24')
        s3 = self.addHost('s3', ip='10.0.0.3/24')
        s4 = self.addHost('s4', ip='10.0.0.4/24')
        
        # Routers (as switches)
        r1 = self.addSwitch('r1', cls=OVSKernelSwitch)
        
        # Links
        self.addLink(s1, r1, bw=10, delay='0.1ms')
        self.addLink(s2, r1, bw=10, delay='0.1ms')
        self.addLink(s3, r1, bw=10, delay='0.1ms')
        self.addLink(s4, r1, bw=10, delay='0.1ms')