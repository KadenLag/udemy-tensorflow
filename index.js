require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true, // shuffle dataset
    splitTest: 50, // record to split for testing
    dataColumns: ['horsepower', 'weight', 'displacement'], // data columns to use for calculation accuracy
    labelColumns: ['mpg'] // result we are looking for
});

const regression = new LinearRegression(features, labels,  {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10 // setting to 1 is stochastic
});

regression.features.print();

regression.train();
const r2 = regression.test(testFeatures, testLabels); // dev

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: ' Mean Squared Error'
});

console.log('R2 is', r2);

regression.predict([ // add new vehicles with specs (must match datacolumns)
    [120, 2, 380]
]).print();

// (still goes over all data set for all methods)
// gradient descent - all observations
// batch gradient descent - couple of observations
// stochastic gradient descent - one observations at a time