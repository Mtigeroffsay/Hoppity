void name_0 ( wchar_t * data ) {  name_1  ( data ) ; } void name_3 ( ) { wchar_t * data ; wchar_t dataBuffer [ 100 ] ;  data  =  dataBuffer  ; wmemset ( data , L 'A' , 100 - 1 ) ; data [ 100 - 1 ] = L '\0' ;  name_2  ( data ) ; } int main ( int argc char * argv [ ] ) {  srand  ( ( unsigned )   time  ( NULL )  ) ;  name_3  ( ) ; return 0 ; } 