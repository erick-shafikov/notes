<!DOCTYPE html>
<html>
<body>
<script>


let str2 = '07:05:45PM';
let str3 = '12:05:39AM';
let str4 = '12:45:54PM';
let str5 = '12:00:00AM';

function timeConversion(s){

    let hoursPM = +s[0]*10 + +s[1] + 12;

    if((s[0]=='1')&&(s[1]=='2')&&(s[8]=='A')) {
        return `00:${s.slice(3,8)}`;
    } else if ((s[0]=='1')&&(s[1]=='2')&&(s[8]=='P')){
        return s.slice(0,8);
    } else if (s[8]=='A') {
        return s.slice(0,8)
    } else if (s[8]=='P'){
        return `${hoursPM}:${s.slice(3,8)}`
    }
}


alert(`${timeConversion(str3)}00:05:39`); 
alert(`${timeConversion(str5)}12:45:54`); 
alert(`${timeConversion(str5)}00:00:00`);

</script>
</body>
</html>