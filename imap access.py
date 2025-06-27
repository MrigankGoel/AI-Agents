# --- CONFIGURATION VARIABLES ---
# Please update these with your details.

# IMAP Server Details
IMAP_SERVER = 'imap.iitb.ac.in'  # IIT Bombay IMAP server
IMAP_PORT = 993                   # Standard IMAP SSL port

# Email Account Credentials
EMAIL_USERNAME = 'XXbYYYY@iitb.ac.in' #Replace with your LDAP address
EMAIL_PASSWORD = 'Insert your webmail access token'  # Access token (same as the one you used while setting up your email client)

# Number of emails to fetch
NUM_EMAILS_TO_FETCH = 5

# --- END OF CONFIGURATION VARIABLES ---

import imaplib
import email
from email.header import decode_header
import csv
import socket # For network-related exceptions
import ssl    # For SSL context
import datetime # For generating timestamped filenames
import pytz     # For timezone handling (IST)

def clean_header_text(header_value):
    """
    Decodes email header text (like Subject, From, To) and handles character sets.
    Returns a clean string.
    """
    if header_value is None:
        return ""
    
    decoded_parts = []
    for part, charset in decode_header(header_value):
        if isinstance(part, bytes):
            try:
                decoded_parts.append(part.decode(charset or 'utf-8', errors='replace'))
            except (UnicodeDecodeError, LookupError): # LookupError for unknown encoding
                decoded_parts.append(part.decode('latin-1', errors='replace')) # Fallback
        else:
            decoded_parts.append(part) # Already a string
    return "".join(decoded_parts)

def get_email_body(msg):
    """
    Extracts the body of the email.
    Prioritizes plain text over HTML. If HTML is used, the full raw HTML is returned.
    """
    body = ""
    if msg.is_multipart():
        # Iterate over email parts
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain" and body == "": # Prioritize plain text
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    body = payload.decode(charset, errors='replace')
                except Exception as e:
                    print(f"  Error decoding text/plain part: {e}")
                    # Try a common fallback if charset is problematic
                    try:
                        body = payload.decode('latin-1', errors='replace')
                    except:
                         body = "Error decoding body (text/plain)"
                # Once plain text is found, we can often stop for the main body.

            elif content_type == "text/html" and body == "": # Fallback to HTML if no plain text found yet
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    # Get the full HTML content, without truncation
                    body = payload.decode(charset, errors='replace') 
                except Exception as e:
                    print(f"  Error decoding text/html part: {e}")
                    try:
                        body = payload.decode('latin-1', errors='replace')
                    except:
                        body = "Error decoding body (text/html)"

    else: # Not a multipart email, just a single part
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                body = payload.decode(charset, errors='replace')
            except Exception as e:
                print(f"  Error decoding single part text/plain: {e}")
                try:
                    body = payload.decode('latin-1', errors='replace')
                except:
                    body = "Error decoding body (single part text/plain)"
        elif content_type == "text/html": 
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                # Get the full HTML content, without truncation
                body = payload.decode(charset, errors='replace')
            except Exception as e:
                print(f"  Error decoding single part text/html: {e}")
                try:
                    body = payload.decode('latin-1', errors='replace')
                except:
                    body = "Error decoding body (single part text/html)"
        else:
            body = f"[Unsupported Content Type: {content_type}]"

    return body.strip() if body else "Body not found or not plain text/HTML"


def main():
    """
    Main function to connect to IMAP, fetch emails, and save to CSV.
    """
    emails_data = []
    mail = None  # Initialize mail connection variable

    # Generate dynamic CSV filename with IST timestamp
    try:
        ist_timezone = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.datetime.now(ist_timezone)
        csv_filename = now_ist.strftime("emails_%Y-%m-%d_%H-%M-%S_IST.csv")
        print(f"Output CSV file will be: {csv_filename}")
    except pytz.exceptions.UnknownTimeZoneError:
        print("Error: Timezone 'Asia/Kolkata' not found. Make sure 'pytz' is installed correctly.")
        print("Falling back to UTC timestamp for filename.")
        now_utc = datetime.datetime.utcnow()
        csv_filename = now_utc.strftime("emails_%Y-%m-%d_%H-%M-%S_UTC.csv")
        print(f"Output CSV file will be: {csv_filename}")
    except Exception as e_time:
        print(f"Error generating timestamped filename: {e_time}")
        print("Falling back to default filename 'emails_fallback.csv'")
        csv_filename = "emails_fallback.csv"


    print(f"Attempting to connect to {IMAP_SERVER} on port {IMAP_PORT}...")
    try:
        # Create an SSL context for a more secure connection
        context = ssl.create_default_context()
        
        # Attempt to set a specific TLS version that might be more compatible
        # with the server, while still aiming for security.
        if hasattr(context, 'minimum_version') and hasattr(ssl, 'TLSVersion') and hasattr(ssl.TLSVersion, 'TLSv1_2'):
            print("Setting minimum TLS version to TLSv1.2")
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        elif hasattr(ssl, 'PROTOCOL_TLSv1_2'): 
             print("Setting SSL protocol to TLSv1.2 (older Python context setup - may not be effective with create_default_context)")
        
        try:
            print("Attempting to set ciphers to 'DEFAULT@SECLEVEL=1' for broader compatibility...")
            context.set_ciphers('DEFAULT@SECLEVEL=1')
        except Exception as e_cipher:
            print(f"Could not set custom ciphers: {e_cipher}")


        # Connect to the server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT, ssl_context=context)
        print("Connected successfully.")

        # Login to your account
        print(f"Logging in as {EMAIL_USERNAME}...")
        mail.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        print("Logged in successfully.")

        # Select the mailbox you want to check (e.g., 'INBOX')
        status, messages_count_raw = mail.select('INBOX')
        if status != 'OK':
            print(f"Error selecting INBOX: {status}")
            return

        messages_count = int(messages_count_raw[0])
        print(f"Total messages in INBOX: {messages_count}")

        if messages_count == 0:
            print("No emails found in INBOX.")
            return

        # Search for all emails in the mailbox
        status, email_ids_raw = mail.search(None, 'ALL') # Get all emails
        if status != 'OK':
            print(f"Error searching for emails: {status}")
            return

        email_id_list = email_ids_raw[0].split() # List of email IDs (bytes)
        
        if not email_id_list:
            print("No email IDs found after search.")
            return

        # Determine the actual number of emails to fetch
        num_to_fetch_actual = min(NUM_EMAILS_TO_FETCH, len(email_id_list))

        if num_to_fetch_actual == 0:
             print("NUM_EMAILS_TO_FETCH is 0 or no emails to process. Fetching none.")
             return
        
        print(f"Fetching the newest {num_to_fetch_actual} email(s)...")

        # Get the slice of email IDs for the newest emails
        ids_to_fetch = email_id_list[-num_to_fetch_actual:]


        for email_id in ids_to_fetch: # Iterate through the selected newest email IDs
            print(f"\nProcessing email ID: {email_id.decode()}...")
            
            # Fetch the email data (RFC822 means full message)
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            if status != 'OK':
                print(f"  Error fetching email ID {email_id.decode()}: {status}")
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    # Parse the email content from bytes
                    msg = email.message_from_bytes(response_part[1])

                    # Decode email headers
                    subject = clean_header_text(msg['subject'])
                    from_ = clean_header_text(msg['from'])
                    to_ = clean_header_text(msg['to'])
                    
                    print(f"  From: {from_}")
                    print(f"  To: {to_}")
                    print(f"  Subject: {subject}")

                    # Get email body
                    body = get_email_body(msg)
                    # print(f"  Body (first 100 chars): {body[:100]}...") # For brevity in console

                    emails_data.append({
                        'From': from_,
                        'To': to_,
                        'Subject': subject,
                        'Body': body,
                        'Raw Email ID': email_id.decode()
                    })
        
        if not emails_data:
            print("No email data was successfully processed.")
            return

        # Write data to CSV file
        print(f"\nWriting {len(emails_data)} emails to {csv_filename}...")
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Raw Email ID', 'From', 'To', 'Subject', 'Body']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for email_info in emails_data:
                    writer.writerow(email_info)
            print(f"Successfully wrote emails to {csv_filename}")
        except IOError as e:
            print(f"Error writing to CSV file {csv_filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during CSV writing: {e}")


    except imaplib.IMAP4.error as e:
        print(f"IMAP Error: {e}")
        print("This could be due to incorrect login credentials, IMAP server issues, or mailbox access problems.")
    except socket.gaierror as e:
        print(f"Network Error: Could not resolve hostname '{IMAP_SERVER}'. Check server address and internet connection. ({e})")
    except socket.error as e: 
        print(f"Socket Connection Error: {e}. Check IMAP server, port, and network.")
    except ssl.SSLError as e:
        print(f"SSL Error: {e}. This might be due to an issue with the server's SSL certificate or your SSL configuration.")
        print("Consider checking the server's SSL/TLS requirements. The script attempted to use TLSv1.2 and a broader cipher set (SECLEVEL=1).")
        print("If this persists, the server may require very specific or outdated SSL/TLS settings not easily compatible with modern Python defaults.")
    except ConnectionRefusedError as e:
        print(f"Connection Refused: {e}. Ensure the IMAP server ({IMAP_SERVER}:{IMAP_PORT}) is correct and accessible.")
    except ImportError:
        print("Error: The 'pytz' library is required for timezone handling but is not installed.")
        print("Please install it by running: pip install pytz")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if mail:
            try:
                print("Logging out and closing connection...")
                mail.logout()
                print("Connection closed.")
            except imaplib.IMAP4.error as e: # Can happen if connection was already dropped
                print(f"Error during logout: {e}")
            except Exception as e:
                print(f"Unexpected error during logout: {e}")


if __name__ == '__main__':
    print("--- Email Fetcher Script ---")
    if IMAP_SERVER == 'imap.example.com' or EMAIL_USERNAME == 'your_email@example.com' or EMAIL_PASSWORD == 'your_email_password':
        print("\nWARNING: Default configuration values are still set.")
        print("Please update IMAP_SERVER, EMAIL_USERNAME, and EMAIL_PASSWORD in the script.\n")
    
    # Check if pytz is available
    try:
        import pytz
    except ImportError:
        print("--------------------------------------------------------------------")
        print("ERROR: The 'pytz' library is required for IST timestamped filenames.")
        print("Please install it by running: pip install pytz")
        print("The script will use UTC timestamps or a fallback filename if pytz is not found.")
        print("--------------------------------------------------------------------")

    main()
    print("--- Script Finished ---")
