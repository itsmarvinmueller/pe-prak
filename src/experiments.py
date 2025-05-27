import time
import argparse
import os
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.net import Mininet
from mininet.topo import Topo

from topologies import BasicTopologie

def runSingleBandwidthExperiment(queue_size=100, udp_bandwidth=15, test_duration=30, iteration=1, result_base_path='./results/tcp_udp_fairness'):
    udp_start_delay = 5
    udp_duration = test_duration - udp_start_delay

    # Create a directory for this specific iteration
    result_path = f'{result_base_path}/queue_{queue_size}_duration_{test_duration}/iteration_{iteration}'
    os.makedirs(result_path, exist_ok=True) 
    
    # Create topology with specified queue size
    topo = BasicTopologie.SimulationTopo(queue_size=queue_size)
    net = Mininet(topo=topo, link=TCLink)
    net.start()
    
    s1, s2, s3, s4 = net.get('s1', 's2', 's3', 's4')
    # s1 -> s3 TCP, s2 -> s4 UDP	
    tcp_server_ip = s3.IP()
    udp_server_ip = s4.IP()

    # Start iperf3 udp server on s4 with port 5202
    s4.cmd(f'iperf3 -s -p 5202 -J > {result_path}/iperf3_udp_server.log 2>&1 &')
    time.sleep(1)

    # Ping for monitoring response times and unfairness between TCP (icmp) and UDP flow
    s1.cmd(f'echo "Start pinging: $(date)" > {result_path}/ping_start.log 2>&1')
    s1.cmd(f'ping -i 1 -w {test_duration + 5} {tcp_server_ip} > {result_path}/ping_result.log 2>&1 &')

    # Start iperf3 udp client on s3
    time.sleep(udp_start_delay)
    s2.cmd(f'iperf3 -c {udp_server_ip} -p 5202 -u -b {udp_bandwidth}M --length 100 -i 5 -t {udp_duration} -J > {result_path}/iperf3_udp.json &')
    
    time.sleep(test_duration + 5)
    
    # kill all processes
    s1.cmd('killall ping 2>/dev/null')
    s4.cmd('killall iperf3 2>/dev/null')
    time.sleep(1)
    
    net.stop()
    
    info(f"\n*** Iteration {iteration} completed with parameters:\n")
    info(f"    - UDP bandwidth: {udp_bandwidth}M\n")
    info(f"    - Queue size: {queue_size} packets\n")
    info(f"    - Test duration: {test_duration} seconds\n")
    info(f"    - Results saved in {result_path}\n")
    
    return result_path