const auth = require('./auth');
const guilinElec = require('./guilinElec');

module.exports = {
    ...auth,
    ...guilinElec
};
