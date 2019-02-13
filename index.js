require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true, // shuffle dataset
    splitTest: 50, // record to split for testing
    dataColumns: ['horsepower'], // data columns to use for calculation accuracy
    labelColumns: ['mpg'] // result we are looking for
});

const regression = new LinearRegression(features, labels,  {
    learningRate: 1,
    iterations: 100
});

regression.features.print();

regression.train();
const r2 = regression.test(testFeatures, testLabels);

console.log('R2 is', r2);
