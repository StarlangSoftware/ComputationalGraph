package ComputationalGraph;

import java.util.ArrayList;
import java.util.List;

public class Class {
    public static void main(String[] args) {
        ArrayList<String> testList = new ArrayList<String>(List.of("test1", "test2", "test3"));
        String s = testList.remove(0);
        System.out.println(testList.toString());
    }

}
