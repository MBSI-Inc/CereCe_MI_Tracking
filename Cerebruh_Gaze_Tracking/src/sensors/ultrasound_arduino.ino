// Define pins
const int trigPin = 9;
const int echoPin = 10;

void setup() {
  // Start serial communication for debugging
  Serial.begin(9600);
  // Set trigPin as output and echoPin as input
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  long duration;
  float distance;

  // Clear trigPin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Send 10us pulse to trigger measurement
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Read echoPin, returns time in microseconds
  duration = pulseIn(echoPin, HIGH);

  // Calculate distance in cm
  distance = duration * 0.0343 / 2;

  // Print the distance
  Serial.println(distance);

  delay(200); // Small delay between readings
}
