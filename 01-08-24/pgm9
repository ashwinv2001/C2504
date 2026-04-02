QUESTION
9.ReadAndFindSumOfDoctorsSalariesTillZero() use while loop 


***pseudocode***


function ReadAndFindSumOfDoctorsSalariesTillZero()
    sum = 0 
    input salary
    while salary != 0
        if (salary < 0) //Invalid salary
            print "Input valid salary"
            input salary
            continue
        end if         
        
        sum += salary
        input salary
    end while 
    print sum 


  ***C# CODE****


class ReadAndFindSumOfDoctorsSalariesTillZerowhileloop
{
 static void ReadAndFindSumOfDoctorsSalariesTillZero()
 {
   int sum = 0 ;
   Console.Write("Enter salary");
   int salary = int.Parse(Console.ReadLine());
    while (salary != 0)
    {
        if (salary < 0) //Invalid salary
        {
          Console.WriteLine("Input valid salary");
          Console.Write("Enter salary");
          salary = int.Parse(Console.ReadLine());
          continue;
        }         
        sum += salary;
        Console.Write("Enter salary");
        salary = int.Parse(Console.ReadLine());
    }
   Console.WriteLine($"sum is : {sum}");
}

 static void Main(string[] args) 
    {
        ReadAndFindSumOfDoctorsSalariesTillZero();
        Console.WriteLine("Press any key to contine...");
        Console.ReadKey();
    }
 }
