#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging

from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.feetech import FeetechMotorsBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ping_motor(port: str, motor_id: int, baudrate: int, protocol_version: int):
    """Pings a single motor ID at a specific baudrate and protocol."""
    logger.info(
        f"Pinging motor ID #{motor_id} on port {port} at {baudrate} baud using protocol {protocol_version}..."
    )
    # We need a dummy motor model for bus initialization. 'sts3215' is fine.
    bus = FeetechMotorsBus(
        port,
        motors={"ping_motor": Motor(motor_id, "sts3215", MotorNormMode.RANGE_0_100)},
        protocol_version=protocol_version,
    )

    if not bus.port_handler.openPort():
        logger.error(f"Failed to open port {port}")
        return

    bus.port_handler.setBaudRate(baudrate)

    model_number = bus.ping(motor_id)

    if model_number:
        logger.info(
            f"SUCCESS: Found motor with ID #{motor_id}. It responded with model number: {model_number}."
        )
    else:
        logger.error(f"FAILURE: No response from motor with ID #{motor_id}.")

    bus.port_handler.closePort()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ping a specific Feetech motor ID.")
    parser.add_argument("port", type=str, help="The serial port (e.g., /dev/ttyUSB0).")
    parser.add_argument("motor_id", type=int, help="The motor ID to ping.")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="Baudrate (default: 1,000,000).")
    parser.add_argument(
        "--protocol",
        type=int,
        default=0,
        choices=[0, 1],
        help="Protocol version (default: 0).",
    )
    args = parser.parse_args()

    ping_motor(args.port, args.motor_id, args.baudrate, args.protocol) 