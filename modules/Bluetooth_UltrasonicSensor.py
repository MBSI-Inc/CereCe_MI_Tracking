from bleak import BleakScanner, BleakClient
import asyncio

DEVICE_NAME = "UltrasonicBLE"
CHAR_UUID = "a4b5735b-6dd9-4680-ac4b-06b4a209bffa"
async def main():
    print("Scanning...")

    devices = await BleakScanner.discover() #Find all devices
    target_address = None

    for d in devices: #Look for device with our respective set name
        if d.name == DEVICE_NAME:
            target_address = d.address
            break

    if target_address is None:
        print("Device not found")
        return

    print(f"Found device at {target_address}") #Finds MAC_address of device 
    
    def handler(sender, data):
        value = data.decode()
        print("Warning value:", value)

        if value == "1":
            print("⚠️ OBJECT DETECTED")
        else:
            print("✅ Clear")


    async with BleakClient(target_address) as client:
        await client.start_notify(CHAR_UUID, handler) #Get notificaiton

        print("Listening for updates...")
        while True: #Runs forever
            await asyncio.sleep(1)

asyncio.run(main())