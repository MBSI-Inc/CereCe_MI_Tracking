from bleak import BleakScanner, BleakClient
import asyncio

DEVICE_NAME = "UltrasonicBLE"
CHAR_UUID = "a4b5735b-6dd9-4680-ac4b-06b4a209bffa"

async def ultrasonic_result(callback_func):
    max_dist = 50; #tune this
    
    print("Scanning...")
    
    devices = await BleakScanner.discover() #Find all devices
    target_address = None

    for d in devices: #Look for device with our respective set name
        if d.name == DEVICE_NAME:
            target_address = d.address
            break

    if target_address is None:
        print("Device not found")
        #Light is off
        return

    print(f"Found device at {target_address}") #Finds MAC_address of device 
    #Light turns on
    
    def handler(sender, data):
        try:
            distance_lhs, distance_rhs = map(float, data.decode().split(",")) #decodes bytes -> data and gets individual values
            #map applies the float function to each number, making the string -> decimal
            if distance_lhs < max_dist or distance_rhs < max_dist:
                print("⚠️ OBJECT DETECTED")
            else:
                print("✅ Clear")
            callback_func(distance_lhs, distance_rhs)
        except Exception as e:
            print(f"Error parsing BLE data: {e}")  


    async with BleakClient(target_address) as client:
        await client.start_notify(CHAR_UUID, handler) #Get notificaiton

        print("Listening for updates...")
        while True: #Runs forever
            await asyncio.sleep(1)
if __name__ == "__main__": #Skips everything after this if not imported
    def dummy_callback(lhs, rhs):
        print(f"LHS: {lhs} | RHS: {rhs}") 
    asyncio.run(ultrasonic_result(dummy_callback))
