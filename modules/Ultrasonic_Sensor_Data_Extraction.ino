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

float lastDistanceLHS = -1;
float lastDistanceRHS = -1;
float changeThreshold = 0.5; // cm - only notify if distance changes by more than this, tune as needed

//#define LED_Bluetooth 8
const int LED_Bluetooth = D8;
const int LED_PIN = D2; 
const int BUTTON_PIN = D5;
//#define LED_PIN 9
//#define BUTTON_PIN 10
byte lastButtonState;
unsigned long lastTimeButtonStateChanged = 0;
unsigned long debounceDuration = 200; //millis
int ledState = LOW;
// See the following for generating UUIDs:
// https://www.uuidgenerator.net/


//Generated UUID with UUID Generator
#define SERVICE_UUID        "62ca07c9-4ade-44c8-8a00-4195bf215b3d"
#define CHARACTERISTIC_UUID "a4b5735b-6dd9-4680-ac4b-06b4a209bffa"
BLECharacteristic *pCharacteristic;
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override {
        digitalWrite(LED_Bluetooth, HIGH); // Turn LED ON when Python connects
        Serial.println("Python Connected!");
    }
    void onDisconnect(BLEServer* pServer) override {
        digitalWrite(LED_Bluetooth, LOW);  // Turn LED OFF when Python disconnects
        Serial.println("Python Disconnected. Restarting advertising...");
        pServer->startAdvertising();        // Keep advertising open for reconnections
    }
};
void setup() {
  Serial.begin(115200); //Double check if nonsense is printed
  delay(1500);
  Serial.println("Starting BLE work!");
  pinMode(trigPinLHS, OUTPUT); //Set up trigPin as an output
  pinMode(echoPinLHS, INPUT); //Set up echoPin as an input
  pinMode(trigPinRHS, OUTPUT);
  pinMode(echoPinRHS, INPUT);

//LED Bluetooth
  pinMode(LED_Bluetooth, OUTPUT);
  digitalWrite(LED_Bluetooth, HIGH);
//LED Power
  pinMode(BUTTON_PIN, INPUT_PULLDOWN);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  delay(2000);
  digitalWrite(LED_PIN, HIGH);
  lastButtonState = digitalRead(BUTTON_PIN); //Get the starting button state

  if (!BLEDevice::init("UltrasonicBLE")) {
    Serial.println("BLE initialization failed!");
    return;
  }

  BLEServer *pServer = BLEDevice::createServer();
 
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID, 
    BLECharacteristic::PROPERTY_READ | //Only read the data 
    BLECharacteristic::PROPERTY_NOTIFY //Only send message if there are changes
    );
  pCharacteristic->setValue("0.0,0.0");
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
{//Logic to determine if Power on/off and light LED if needed
  if (millis() - lastTimeButtonStateChanged >= debounceDuration) { //If difference between right now and last state change is greater than the green line on our diagram. If it's less than debounce is occuring (a.k.a flickering) so don't change.
    byte buttonState = digitalRead(BUTTON_PIN); //Current button state
    if (buttonState != lastButtonState){ //If button state has changed (press/release occured)
      lastTimeButtonStateChanged = millis(); //Get the time since the last time the button state changed
      lastButtonState = buttonState; //Redefine so both are the same again
      Serial.println("Button State has changed!");
      if(buttonState == HIGH){
        Serial.println("HIGH!");
        if (ledState == LOW){
          digitalWrite(LED_PIN, HIGH); //Check
          Serial.println("Light On!");
          ledState = HIGH;
        }
        else if (ledState == HIGH){
          digitalWrite(LED_PIN, LOW); //Check
          ledState = LOW;
        }
      }
    }
  }
  if (ledState == HIGH) {//Ultrasonic Measurements only occur if 'power' is on
    digitalWrite(trigPinLHS, LOW); 
    delayMicroseconds(2);
    digitalWrite(trigPinLHS, HIGH);//Transmit signal
    delayMicroseconds(10);
    digitalWrite(trigPinLHS,LOW);
    
    //Records how long it takes for signal to be received
    durationLHS = pulseIn(echoPinLHS, HIGH); //echoPin is HIGH when sound waves hit the receiver
    distanceLHS = (durationLHS*0.0343)/2;
    delayMicroseconds(60);
  
    
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

    // Build a simple "LHS,RHS" string payload, e.g. "12.34,56.78"
    char payload[32]; //preallocation
    snprintf(payload, sizeof(payload), "%.2f,%.2f", distanceLHS, distanceRHS);
    pCharacteristic->setValue(payload);

    // Only notify when a distance actually changed by more than the threshold,
    // so we're not spamming BLE notifications every loop with noise-level jitter.
    bool changed = (fabs(distanceLHS - lastDistanceLHS) > changeThreshold) ||
                   (fabs(distanceRHS - lastDistanceRHS) > changeThreshold);

    if (changed) {
      pCharacteristic->notify();
      lastDistanceLHS = distanceLHS;
      lastDistanceRHS = distanceRHS;
      Serial.print("Notified distances: ");
      Serial.println(payload);
    }

    delay(200); // shortened from 2000ms so Python gets more responsive updates; tune as needed
  }
}
