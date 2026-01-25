const auth = require('./api/auth');
const student = require('./api/student');

module.exports = {
    ...auth,
    ...student
};
