#!usr/bin/ven python3
# -*- coding: utf-8 -*-

import logging, sys
import time
from collections import OrderedDict

from web3 import Web3
from pymongo import MongoClient, ASCENDING

logging.basicConfig(stream=sys.stdout, format="%(levelname)s: %(asctime)s: %(message)s", level=logging.INFO,
                    datefmt='%a %d %b %Y %H:%M:%S')
log = logging.getLogger(__name__)
import faulthandler
faulthandler.enable()

import binascii
from concurrent.futures import ThreadPoolExecutor
from config import ARCHIVE_NODE_ENDPOINTS
from typing import Tuple


class ITRBuilder:
    def __init__(self, start_block: int, end_block: int):
        self._start_block = start_block
        self._end_block = end_block
        self._w3 = None
        self._reset_endpoint()
        pass

    def _reset_endpoint(self, ) -> None:
        import random

        url = ARCHIVE_NODE_ENDPOINTS[random.randint(0, len(ARCHIVE_NODE_ENDPOINTS) - 1)]

        if url.startswith("http"):
            self._w3 = Web3(Web3.HTTPProvider(url))
        elif url.startswith("ws"):
            self._w3 = Web3(Web3.WebsocketProvider(url))
        else:
            self._w3 = Web3(Web3.IPCProvider(url))

        return


    def _get_trace(self, ) -> Tuple[int, str, list]:
        import requests
        import json
        from tqdm import tqdm

        '''
        pbar = tqdm(desc="Progress: ", total=self._end_block - self._start_block + 1,
                    unit='B', unit_scale=True,
                    unit_divisor=1024,
                    miniters=1)

        pbar.set_postfix_str(f"Working block: ".format(self._start_block))
        '''

        for working_block in range(self._start_block, self._end_block + 1):
            '''
            pbar.set_postfix_str(f"Working block: ".format(working_block))
            pbar.update(1)
            '''
            while True:
                try:
                    tx_num = self._w3.eth.get_block_transaction_count(working_block)
                    break
                except Exception as e:
                    log.error("get_tx_count error: {}, endpoint: {}".format(e, self._w3.provider.endpoint_uri))
                    self._reset_endpoint()
                    continue

            working_txs = []
            for working_tx_num in range(tx_num):
                while True:
                    try:
                        working_tx = self._w3.eth.get_transaction_by_block(working_block, working_tx_num)
                        break
                    except Exception as e:
                        log.error("get_tx error: {}, endpoint: {}".format(e, self._w3.provider.endpoint_uri))
                        self._reset_endpoint()
                        continue

                if working_tx["to"] is None:
                    continue
                # doc = self._db_client["ethereum"]["contract_account_v2"].find_one(
                #     {"account_lower": working_tx["to"].lower()})
                # if not doc:
                # https://app.bountysource.com/issues/49372248-debug_tracetransaction-is-not-implemented
                # https://geth.ethereum.org/docs/interacting-with-geth/rpc/ns-debug
                # https://www.quicknode.com/docs/ethereum/debug_traceTransaction
                #
                while True:
                    try:
                        bin_code = self._w3.eth.get_code(working_tx["to"])
                        break
                    except Exception as e:
                        log.error("get_code error: {}, endpoint: {}".format(e, self._w3.provider.endpoint_uri))
                        self._reset_endpoint()
                        continue

                if bin_code.hex() != '0x':
                    while True:
                        try:
                            headers = {'content-type': 'application/json'}
                            payload = {
                                "method": "debug_traceTransaction",
                                "params": ["{}".format(working_tx["hash"].hex())],
                                # "params": ["{}".format(working_tx["hash"].hex()), {"tracer": "callTracer"}],
                                # "params": ["{}".format(working_tx["hash"].hex()), {"tracer": "prestateTracer"}],
                                "jsonrpc": "2.0",
                                "id": 1,
                            }
                            r = requests.post(self._w3.provider.endpoint_uri, data=json.dumps(payload),
                                headers=headers)
                        except Exception as e:
                            log.error("get_trace error: {}, endpoint: {}".format(e, self._w3.provider.endpoint_uri))
                            self._reset_endpoint()
                            continue

                        if r.status_code != 200:
                            self._reset_endpoint()
                            continue

                        try:
                            json_r = r.json()
                        except Exception as e:
                            log.error("get_json error: {}, endpoint: {}".format(e, self._w3.provider.endpoint_uri))
                            self._reset_endpoint()
                            continue

                        # {'jsonrpc': '2.0', 'error': {'code': -32601, 'message': 'the method debug_traceTransaction does not exist/is not available', 'data': None}, 'id': 1}
                        if 'error' in json_r.keys():
                            self._reset_endpoint()
                            continue
                        log.info("working archive node: {}".format(self._w3.provider.endpoint_uri))
                        break

                    # build a pseudo `call`
                    import re
                    trace = []
                    trace.append(
                        {
                            "pc": -1,
                            "op": "CALL",
                            "gas": working_tx["gas"],
                            "gasPrice": working_tx["gasPrice"],     # additional gas price
                            "depth": 0,
                            "stack": [
                                str(hex(len(working_tx["input"]))),
                                "0x0",
                                str(hex(working_tx["value"])),
                                working_tx["to"],
                                str(hex(working_tx["gas"]))
                            ],
                            "memory": re.findall(r'.{64}', working_tx["input"].hex()[2:])
                        }
                    )

                    # trace.append)
                    trace += json_r['result']['structLogs']
                    '''
                    trace["root"]["gas"] = working_tx["gas"]                                        # abnormal tx costs more gas
                    trace["root"]["gasPrice"] = working_tx["gasPrice"]                              # front runner pays higher gas price
                    if "maxPriorityFeePerGas" in working_tx.keys():                                 # tips for miner, depend on type 0 or 2
                        trace["root"]["maxPriorityFeePerGas"] = working_tx["maxPriorityFeePerGas"]
                    if "maxFeePerGas" in working_tx.keys():
                        trace["root"]["maxFeePerGas"] = working_tx["maxFeePerGas"]
                    trace["root"]["input"] = working_tx["input"].hex()[:10]                         # should I break down the `input`?
                    trace["root"]["to"] = working_tx["to"]                                          # an identicator of target contract
                    trace["root"]["value"] = working_tx["value"]                                    # the `eth` payed.
                    '''
                    yield working_block, working_tx["hash"].hex(), trace

        pass


    def _get_data_list(self, memory: list, offset: int, length: int) -> list:
        import re

        dataz = ""
        for data in memory:
            dataz += data
        dataz = dataz[offset * 2: offset * 2 + length * 2]

        return re.findall(r'.{64}', dataz)

    def get_data_str(self, memory: list, offset: int, length: int) -> str:
        import re

        dataz = ""
        for data in memory:
            dataz += data
        return dataz[offset * 2: offset * 2 + length * 2]


    def _rebuild_trace(self, trace: dict) -> None:
        callers = []
        callers.append(trace[0])
        for i in range(1, len(trace)):
            inst = trace[i]
            if inst["op"] in ['CALL', 'STATICCALL', 'DELEGATECALL', 'CALLCODE']:
                log.info("\"pc\": {}, op: {}, depth: {}".format(inst["pc"], inst["op"], inst["depth"]))
                # https://malimacode.hashnode.dev/how-to-call-an-eoas-in-a-smart-contract
                if inst["depth"] == trace[i + 1]["depth"]:
                    inst["from"] = callers[-1]["stack"][-2]
                    inst["ret"] = []
                else:
                    inst["from"] = callers[-1]["stack"][-2]
                    callers.append(inst)
            elif inst["op"] == 'RETURN':
                log.info("\"pc\": {}, op: {}, depth: {}".format(inst["pc"], inst["op"], inst["depth"]))
                caller = callers.pop()
                if "memory" not in inst.keys():
                    caller["ret"] = []
                else:
                    caller["ret"] = self._get_data_list(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-1], 16),
                        length=int(inst["stack"][-2], 16))
            elif inst["op"] == 'STOP':
                log.info("\"pc\": {}, op: {}, depth: {}".format(inst["pc"], inst["op"], inst["depth"]))
                caller = callers.pop()
                caller["ret"] = []
            #         https://blog.chain.link/events-and-logging-in-solidity/
            elif "LOG" in inst["op"]:
                inst["contract_addr"] = callers[-1]["stack"][-2]

                pass

    def _traverse_scheme_2(self, trace: list) -> dict:
        '''
        equivalent depth-first search result, do nothing, just return the inst one by one
        :param trace:
        :return:
        '''
        for inst in trace:
            yield inst

    def _traverse_scheme_2(self, trace: list) -> dict:
        '''
        equivalent width-first search result
        :param trace:
        :return:
        '''

        depth = -1
        while True:
            found = False
            depth += 1
            for inst in trace:
                if inst["depth"] == depth:
                    yield inst
                    found = True
            if not found:
                break

    def _build_itr(self, trace) -> str:
        retMe = ""
        self._rebuild_trace(trace)
        '''
        import json
        with open("tmp.dat", "w") as fwh:
            json.dump(trace, fwh, indent=2)
        '''

        for inst in self._traverse_scheme_2(trace):
            # [START], [CALL], from, to, function hash, gas, value, [INs], input1, type, input1, value, ..., [OUTS], output type, out value, ... [END]
            if inst["op"] in ['CALL', 'CALLCODE']:
                retMe += "[START], [CALL], {}, {}, {}, {}, {}, [INs], {}, [OUTs], {}, [END]".format(
                    inst["from"] if inst["depth"] != 0 else "NONE",
                    inst["stack"][-2],
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-4], 16),
                        length=4
                    ),
                    int(inst['stack'][-1], 16),
                    int(inst['stack'][-3], 16),
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-4], 16) + 4,
                        length=int(inst["stack"][-5], 16) - 4
                    ),
                    "".join(inst["ret"])
                )
            # [START], [CALL], from, to, function hash, gas, 0, [INs], input1, type, input1, value, ..., [OUTS], output type, out value, ... [END]
            elif inst["op"] in ['STATICCALL', 'DELEGATECALL']:
                retMe += "[START], [CALL], {}, {}, {}, {}, 0, [INs], {}, [OUTs], {}, [END]".format(
                    inst["from"] if inst["depth"] != 0 else "NONE",
                    inst["stack"][-2],
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-4], 16),
                        length=4
                    ),
                    int(inst['stack'][-1], 16),
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-4], 16) + 4,
                        length=int(inst["stack"][-5], 16) - 4
                    ),
                    "".join(inst["ret"])
                )

            # SLOAD:read
            elif inst['op'] == 'SLOAD':
                retMe += "[START], [STATE], read, {}, {}, [END]".format(
                    inst["stack"][-1],
                    inst["storage"][
                        '{:0>64x}'.format(int(inst["stack"][-1], 16))
                    ]
                )
            # SSTORE:write
            elif inst['op'] == 'SSTORE':
                retMe += "[START], [STATE], write, {}, {}, [END]".format(
                    inst["stack"][-1],
                    inst["stack"][-2],
                )

            elif inst['op'] == 'LOG0':
                retMe += "[START], [LOG], {}, {}, [END]".format(
                    inst["contract_addr"],
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-1], 16),
                        length=int(inst["stack"][-2], 16)
                    ),
                )

            elif inst['op'] == 'LOG1':
                retMe += "[START], [LOG], {}, {}, {}, [END]".format(
                    inst["contract_addr"],
                    inst["stack"][-3],
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-1], 16),
                        length=int(inst["stack"][-2], 16)
                    ),
                )

            elif inst['op'] == 'LOG2':
                retMe += "[START], [LOG], {}, {}, {}, {}, [END]".format(
                    inst["contract_addr"],
                    inst["stack"][-3],
                    inst["stack"][-4],
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-1], 16),
                        length=int(inst["stack"][-2], 16)
                    ),
                )

            elif inst['op'] == 'LOG3':
                retMe += "[START], [LOG], {}, {}, {}, {}, {}, [END]".format(
                    inst["contract_addr"],
                    inst["stack"][-3],
                    inst["stack"][-4],
                    inst["stack"][-5],
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-1], 16),
                        length=int(inst["stack"][-2], 16)
                    ),
                )

            elif inst['op'] == 'LOG4':
                retMe += "[START], [LOG], {}, {}, {}, {}, {}, {}, [END]".format(
                    inst["contract_addr"],
                    inst["stack"][-3],
                    inst["stack"][-4],
                    inst["stack"][-5],
                    inst["stack"][-6],
                    self.get_data_str(
                        memory=inst["memory"],
                        offset=int(inst["stack"][-1], 16),
                        length=int(inst["stack"][-2], 16)
                    ),
                )

        return retMe

    def go(self, ):
        import json

        for working_block, working_tx_hash, working_trace in self._get_trace():
            log.info("dumping block: {}, transaction hash: {}.".format(working_block, working_tx_hash))
            with open("{}-{}.dat".format(working_block, working_tx_hash), "w") as fwh:
                json.dump(working_trace, fwh, indent=2)

            with open("{}-{}.dat".format(working_block, working_tx_hash)) as frh:
                trace = json.load(frh)

            try:
                token = itr_builder._build_itr(trace)

                with open("{}-{}-token.dat".format(working_block, working_tx_hash), "w") as fwh:
                    fwh.write(token)
            except:
                log.error("revert or out of gas tx, hash: {}.".format(working_tx_hash))

if __name__ == '__main__':
    itr_builder = ITRBuilder(start_block=18689700, end_block=18689710)
    itr_builder.go()

