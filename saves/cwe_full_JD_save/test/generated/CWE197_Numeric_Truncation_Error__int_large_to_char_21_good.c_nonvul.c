int name_2 ( int data ) { if ( name_1 ) {  data  =   CHAR_MAX  +  5   ; } return data ; } void name_3 ( ) { int data ;  data  =  -  1   ;  name_1  =  1  ;  data  =   name_2  ( data )  ; { char  charData  =  ( char )  data   ;  printHexCharLine  ( charData ) ; } } int main ( int argc char * argv [ ] ) {  srand  ( ( unsigned )   time  ( NULL )  ) ;  name_3  ( ) ; return 0 ; } 