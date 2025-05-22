@echo off
echo Building project...
call mvn clean install

echo.
echo Running Stock Classifier...
java -cp target/ai-stock-classifier-1.0-SNAPSHOT-shaded.jar com.stockclassifier.StockClassifier

pause 