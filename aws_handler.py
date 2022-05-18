import random
import requests
import datetime

API_KEY = "eXqSD13oxn2MMjDiiSNs223IJvxC5P0C6Q2fCwe7";
COMPANY = "Knightec";

def get_headers(api_key):
    return {
        "x-api-key" : api_key
    };

def add_entry_event(total_in, total_out, company=COMPANY, api_key=API_KEY):
    if company is None:
        print("You must provide company to perform this operation");
        return;

    id = get_next_id(company);
    response = requests.post(
        'https://nj910rabp4.execute-api.us-east-1.amazonaws.com/Beta', 
        json = {
            'event_id': id, 
            'company': company, 
            'total_in': total_in, 
            'total_out': total_out
        },
        headers = get_headers(api_key)
    );

    print(response.text);

def delete_entry_event(api_key, company, event_id):
    response = requests.delete(
        url = 'https://nj910rabp4.execute-api.us-east-1.amazonaws.com/Beta', 
        json = {
            'company': company, 
            'event_id': event_id
        },
        headers = get_headers(api_key)
    );

    print(response.text);

def get_entry_events(company):
    payload = {
        'company': company
    };
    response = requests.get(
        url = 'https://nj910rabp4.execute-api.us-east-1.amazonaws.com/Beta',
        params = payload
    );

    # print(response.text);

    return response.json();

def get_next_id(company):
    entry_events = get_entry_events(company);
    next_id = len(entry_events["body"]);

    return next_id;

def generate_random_date():
    start_date = datetime.date(2022, 1, 1);
    now = datetime.datetime.now();
    end_date = datetime.date(now.year, now.month, now.day);

    time_between_dates = end_date - start_date;
    days_between_dates = time_between_dates.days;
    random_number_of_days = random.randrange(days_between_dates);
    random_date = start_date + datetime.timedelta(days=random_number_of_days);

    return random_date;

def format_time(input):
    if input < 10:
        input = "0" + str(input);
    else:
        input = str(input);
    
    return input;

def generate_random_time_of_day():
    hour = format_time(random.randint(6, 19));
    minute = format_time(random.randint(0, 60));
    second = format_time(random.randint(0, 60));

    return hour + ":" + minute + ":" + second;

def populate_dynamodb_table(api_key, iterations, company):
    i = 0;
    while(i < iterations):
        id = get_next_id(company);
        total_in = random.randint(0, 10);
        total_out = random.randint(0, 10);
        random_date = generate_random_date();
        random_weekday = random_date.strftime('%A');
        random_time = generate_random_time_of_day();

        response = requests.post(
            url = 'https://nj910rabp4.execute-api.us-east-1.amazonaws.com/Beta', 
            json = {
                'event_id': id, 
                'company': company, 
                'event_date': str(random_date), 
                'event_time': random_time, 
                'event_weekday': random_weekday, 
                'total_in': total_in, 
                'total_out': total_out
            },
            headers = get_headers(api_key)
        );

        if response.ok:
            print(i);
            i += 1;
            continue;
        else:
            break;

# def main():
    # Code tests
    #       |   
    #       v
    # total_in = random.randint(0, 10);
    # total_out = random.randint(0, 10);
    # add_entry_event(API_KEY, COMPANY, total_in, total_out);
    # delete_entry_event(API_KEY, COMPANY, 0);
    # get_entry_events(COMPANY);
    # print(f"Next id is: {get_next_id(COMPANY)}");
    # print(f"Random date: {generate_random_date()}");
    # print(f"Random time of day: {generate_random_time_of_day()}");

    # Inserting random entry events into the dynamodb table
    #       |   
    #       v
    # populate_dynamodb_table(API_KEY, 999, COMPANY);
    # print("Done!");
    
# if __name__ == "__main__":
    # main();