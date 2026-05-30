/*
    Based on Neil Kolban example for IDF: https://github.com/nkolban/esp32-snippets/blob/master/cpp_utils/tests/BLE%20Tests/SampleServer.cpp
    Ported to Arduino ESP32 by Evandro Copercini
    updates by chegewara
*/

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>


const int trigPinLHS = D0; //define pin that trig is connected to
const int echoPinLHS = D1; //define pin that echo is connected to 
float durationLHS, distanceLHS; //float variables to time taken for soundwave to travel to object and back & how far away object is

const int trigPinRHS = D3; //define pin that trig is connected to
const int echoPinRHS = D4; //define pin that echo is connected to 
float durationRHS, distanceRHS; //float variables to time taken for soundwave to travel to object and back & how far away object is
float max_dist = 15;
bool warning = false;
bool lastWarning = false;
// See the following for generating UUIDs:
// https://www.uuidgenerator.net/


//Generated UUID with UUID Generator
#define SERVICE_UUID        "62ca07c9-4ade-44c8-8a00-4195bf215b3d"
#define CHARACTERISTIC_UUID "a4b5735b-6dd9-4680-ac4b-06b4a209bffa"
BLECharacteristic *pCharacteristic;

void setup() {
  Serial.begin(115200); //Double check if nonsense is printed
  Serial.println("Starting BLE work!");
  pinMode(trigPinLHS, OUTPUT); //Set up trigPin as an output
  pinMode(echoPinLHS, INPUT); //Set up echoPin as an input
  pinMode(trigPinRHS, OUTPUT);
  pinMode(echoPinRHS, INPUT);

  if (!BLEDevice::init("UltrasonicBLE")) {
    Serial.println("BLE initialization failed!");
    return;
  }

  BLEServer *pServer = BLEDevice::createServer();
  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID, 
    BLECharacteristic::PROPERTY_READ | //Only read the data 
    BLECharacteristic::PROPERTY_NOTIFY
    );
  pCharacteristic->setValue(warning);
  pService->start();


  // BLEAdvertising *pAdvertising = pServer->getAdvertising();  // this still is working for backward compatibility
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  // functions that help with iPhone connections issue
  pAdvertising->setMaxPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("Characteristic defined! Now you can read it in your phone!");
}


void loop()
{
  digitalWrite(trigPinLHS, LOW); 
  delayMicroseconds(2);
  digitalWrite(trigPinLHS, HIGH);//Transmit signal
  delayMicroseconds(10);
  digitalWrite(trigPinLHS,LOW);
 	
  //Records how long it takes for signal to be received
  durationLHS = pulseIn(echoPinLHS, HIGH); //echoPin is HIGH when sound waves hit the receiver
  distanceLHS = (durationLHS*0.0343)/2;
  delayMicroseconds(30);
 
  
  digitalWrite(trigPinRHS, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPinRHS, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPinRHS, LOW);
  
  durationRHS = pulseIn(echoPinRHS, HIGH);
  distanceRHS = (durationRHS*0.0343)/2;
  Serial.print("DistanceLHS: ");
  Serial.println(distanceLHS);
  Serial.print("DistanceRHS: ");
  Serial.println(distanceRHS);
  if (distanceRHS < max_dist || distanceLHS < max_dist){
    warning = true;
  } else{
    warning = false;
  }
  Serial.print("Warning: ");
  Serial.println(warning);
  Serial.print("Last Warning: ");
  Serial.println(lastWarning);
  
  if (warning == true) {
      pCharacteristic->setValue("1");
  } else {
      pCharacteristic->setValue("0");
  }

if (warning != lastWarning) {
  pCharacteristic->notify();
  lastWarning = warning;
  Serial.print("Change in Warning: ");
  Serial.println(warning);
  
}
  delay(2000);
}
