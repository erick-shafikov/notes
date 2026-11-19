function gradingStudents(grades) {
  let results = grades.map((grade) => {
    if (grade < 38) {
      return grade;
    } else if ((grade + 1) % 5 === 0) {
      return grade + 1;
    } else if ((grade + 2) % 5 === 0) {
      return grade + 2;
    } else {
      return grade;
    }
  });

  return results;
}

alert(gradingStudents([4, 38, 67, 75]));
